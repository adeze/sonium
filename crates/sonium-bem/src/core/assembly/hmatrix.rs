//! H-matrix assembly with Hybrid Cross Approximation (HCA)
//!
//! Provides hierarchical matrix compression for BEM systems. Far-field blocks
//! are approximated as low-rank `U V^T` factorisations using interpolation-seeded
//! HCA by default, while near-field blocks are stored as dense matrices.
//!
//! Combined with iterative solvers (BiCGSTAB/GMRES), this enables O(N log N)
//! assembly and O(N log N) per iteration for the matrix-vector product.

use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
#[cfg(feature = "cuda")]
use sonium_compute::{
    CudaBackend, CudaDenseBlockC64Input, CudaHMatrixC64, CudaHMatrixWavenumberUpdateC64,
    CudaLowRankBlockC64Input, MemoryPool,
};
use sonium_solvers::dense::DenseMatrix;
use std::cell::RefCell;

use crate::core::assembly::common::{get_bc_type_and_value, get_element_coords};
use crate::core::integration::{
    regular_integration_fixed_order, regular_integration_fixed_order_wavenumber_derivative,
    regular_integration_linear, regular_integration_linear_fixed_order_wavenumber_derivative,
    regular_integration_linear_fixed_order_wavenumber_derivative_with_normals,
    regular_integration_linear_with_normals, regular_integration_wavenumber_derivative,
    regular_integration_with_normals, singular_integration_curved, singular_integration_linear,
    singular_integration_linear_curved, singular_integration_linear_curved_wavenumber_derivative,
    singular_integration_linear_wavenumber_derivative, singular_integration_wavenumber_derivative,
};
use crate::core::parallel::{
    BEM_BLOCKS_PER_THREAD, BEM_ENTRIES_PER_THREAD, gated_enumerate_filter_map, gated_flat_map,
    gated_map, use_parallel,
};
#[cfg(test)]
use crate::core::types::BoundaryCondition;
use crate::core::types::IntegrationResult;
use crate::core::types::{BasisOrder, Cluster, Element, PhysicsParams};
use sonium_solvers::traits::LinearOperator;

/// Failure while assembling an analytic H-matrix material derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixDerivativeError {
    /// A derivative entry was requested without a positive quadrature order.
    InvalidQuadratureOrder,
    /// A matrix block references a DOF with no owning surface element.
    MissingDofOwner {
        /// Global DOF without an owning element.
        dof: usize,
    },
    /// A scalar derivative assembly parameter is not finite.
    InvalidParameter {
        /// Name of the invalid parameter group.
        name: &'static str,
    },
}

impl std::fmt::Display for MatrixDerivativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidQuadratureOrder => {
                write!(f, "wavenumber derivative quadrature order must be positive")
            }
            Self::MissingDofOwner { dof } => {
                write!(f, "wavenumber derivative DOF {dof} has no owning element")
            }
            Self::InvalidParameter { name } => {
                write!(f, "matrix derivative parameter {name} must be finite")
            }
        }
    }
}

impl std::error::Error for MatrixDerivativeError {}

/// Failure while applying a compressed H-matrix to an external vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HMatrixOperatorError {
    /// The input vector does not match the square operator dimension.
    InputLength {
        /// Required operator dimension.
        expected: usize,
        /// Supplied vector length.
        actual: usize,
    },
    /// A caller-provided output buffer has the wrong dimension.
    OutputLength {
        /// Required operator dimension.
        expected: usize,
        /// Supplied buffer length.
        actual: usize,
    },
    /// An input block has the wrong number of rows for this square operator.
    BlockInputRows {
        /// Required number of input rows.
        expected: usize,
        /// Supplied number of input rows.
        actual: usize,
    },
    /// An output block has the wrong number of rows for this square operator.
    BlockOutputRows {
        /// Required number of output rows.
        expected: usize,
        /// Supplied number of output rows.
        actual: usize,
    },
    /// Input and output blocks have different numbers of right-hand sides.
    BlockColumnCount {
        /// Number of input right-hand sides.
        input: usize,
        /// Number of output right-hand sides.
        output: usize,
    },
    /// An input or generated output contains a non-finite value.
    NonFiniteValue {
        /// Index of the non-finite value.
        index: usize,
    },
}

impl std::fmt::Display for HMatrixOperatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputLength { expected, actual } => {
                write!(f, "H-matrix input has length {actual}, expected {expected}")
            }
            Self::OutputLength { expected, actual } => {
                write!(
                    f,
                    "H-matrix output has length {actual}, expected {expected}"
                )
            }
            Self::BlockInputRows { expected, actual } => write!(
                f,
                "H-matrix block input has {actual} rows, expected {expected}"
            ),
            Self::BlockOutputRows { expected, actual } => write!(
                f,
                "H-matrix block output has {actual} rows, expected {expected}"
            ),
            Self::BlockColumnCount { input, output } => write!(
                f,
                "H-matrix block input/output column counts differ: {input} vs {output}"
            ),
            Self::NonFiniteValue { index } => {
                write!(f, "H-matrix vector value at index {index} is non-finite")
            }
        }
    }
}

impl std::error::Error for HMatrixOperatorError {}

/// H-matrix action to validate against an independently assembled reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HMatrixAction {
    /// Forward product `A*x`.
    Forward,
    /// Algebraic transpose product `Aᵀ*x`.
    Transpose,
    /// Hermitian transpose product `Aᴴ*x`.
    Hermitian,
}

/// Failure while comparing a compressed H-matrix action with reference data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HMatrixAccuracyError {
    /// At least one deterministic probe is required.
    EmptyProbeSet,
    /// The number of reference outputs must match the number of probes.
    ProbeCountMismatch {
        /// Number of input probe vectors.
        probes: usize,
        /// Number of independently assembled reference vectors.
        references: usize,
    },
    /// The compressed operator rejected a probe.
    Operator(HMatrixOperatorError),
    /// A reference output has the wrong dimension.
    ReferenceLength {
        /// Probe/reference index.
        probe: usize,
        /// Expected output length.
        expected: usize,
        /// Actual output length.
        actual: usize,
    },
    /// A reference output contains a non-finite value.
    NonFiniteReference {
        /// Probe/reference index.
        probe: usize,
        /// Index of the non-finite value.
        index: usize,
    },
    /// The relative error calculation produced a non-finite value.
    NonFiniteError {
        /// Probe/reference index.
        probe: usize,
    },
}

impl std::fmt::Display for HMatrixAccuracyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProbeSet => {
                write!(f, "H-matrix accuracy check requires at least one probe")
            }
            Self::ProbeCountMismatch { probes, references } => write!(
                f,
                "H-matrix accuracy check received {probes} probes and {references} references"
            ),
            Self::Operator(error) => write!(f, "H-matrix probe action failed: {error}"),
            Self::ReferenceLength {
                probe,
                expected,
                actual,
            } => write!(
                f,
                "H-matrix reference output {probe} has length {actual}, expected {expected}"
            ),
            Self::NonFiniteReference { probe, index } => write!(
                f,
                "H-matrix reference output {probe} has non-finite value at index {index}"
            ),
            Self::NonFiniteError { probe } => {
                write!(
                    f,
                    "H-matrix relative action error for probe {probe} is non-finite"
                )
            }
        }
    }
}

impl std::error::Error for HMatrixAccuracyError {}

/// Matrix-entry kernel used by the shared value and derivative H-matrix
/// builders. Keeping the mode at the ACA boundary makes the derivative path
/// use exactly the same pivot rules and block scheduling as the production
/// value path.
#[derive(Debug, Clone, Copy)]
enum MatrixEntryMode {
    Value {
        gamma: Complex64,
        tau: Complex64,
        beta: Complex64,
        sign: f64,
    },
    WavenumberDerivative {
        gamma: Complex64,
        tau: Complex64,
        beta: Complex64,
        beta_derivative: Complex64,
        sign: f64,
        gauss_order: usize,
    },
    /// Fixed-geometry derivative with respect to the medium density.
    DensityDerivative {
        gamma: Complex64,
        tau: Complex64,
        beta: Complex64,
        sign: f64,
    },
}

impl MatrixEntryMode {
    fn validate(self) -> Result<(), MatrixDerivativeError> {
        match self {
            Self::Value { .. } => Ok(()),
            Self::WavenumberDerivative {
                sign,
                beta_derivative,
                gauss_order,
                ..
            } => {
                if gauss_order == 0 {
                    return Err(MatrixDerivativeError::InvalidQuadratureOrder);
                }
                if !sign.is_finite()
                    || !beta_derivative.re.is_finite()
                    || !beta_derivative.im.is_finite()
                {
                    return Err(MatrixDerivativeError::InvalidParameter {
                        name: "sign and beta derivative",
                    });
                }
                Ok(())
            }
            Self::DensityDerivative { sign, .. } => {
                if !sign.is_finite() {
                    return Err(MatrixDerivativeError::InvalidParameter { name: "sign" });
                }
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        self,
        source_index: usize,
        field_index: usize,
        source_elem: &Element,
        field_elem: &Element,
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Complex64 {
        match self {
            Self::Value {
                gamma,
                tau,
                beta,
                sign,
            } => compute_matrix_entry(
                source_index,
                field_index,
                source_elem,
                field_elem,
                nodes,
                physics,
                gamma,
                tau,
                beta,
                sign,
            ),
            Self::WavenumberDerivative {
                gamma,
                tau,
                beta,
                beta_derivative,
                sign,
                gauss_order,
            } => compute_matrix_entry_wavenumber_derivative(
                source_index,
                field_index,
                source_elem,
                field_elem,
                nodes,
                physics,
                gamma,
                tau,
                beta,
                beta_derivative,
                sign,
                gauss_order,
            )
            .expect("validated H-matrix derivative entry parameters"),
            Self::DensityDerivative {
                gamma,
                tau,
                beta,
                sign,
            } => compute_matrix_entry_density_derivative(
                source_index,
                field_index,
                source_elem,
                field_elem,
                nodes,
                physics,
                gamma,
                tau,
                beta,
                sign,
            ),
        }
    }

    /// Evaluate an entry addressed by global source/field DOFs.
    ///
    /// The original H-matrix builder used one element index per row and
    /// column, which is valid for constant panels only.  Higher-order panels
    /// have several collocation rows and basis-weighted columns per element.
    /// Keep the constant path byte-for-byte equivalent, but route multi-DOF
    /// layouts through the basis-aware entry evaluator.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_dof(
        self,
        source_dof: usize,
        field_dof: usize,
        source_index: usize,
        field_index: usize,
        source_local: usize,
        field_local: usize,
        source_elem: &Element,
        field_elem: &Element,
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Complex64 {
        let curved_normals = shared_proxy_curved_normals(source_elem, field_elem);
        let source_local = decode_dof_proxy_group(source_elem.group)
            .map(|(_, local)| local)
            .unwrap_or(source_local);
        let field_local = decode_dof_proxy_group(field_elem.group)
            .map(|(_, local)| local)
            .unwrap_or(field_local);
        if source_elem.dof_addresses.len() == 1
            && field_elem.dof_addresses.len() == 1
            && decode_dof_proxy_group(source_elem.group).is_none()
            && decode_dof_proxy_group(field_elem.group).is_none()
        {
            return self.evaluate(
                source_index,
                field_index,
                source_elem,
                field_elem,
                nodes,
                physics,
            );
        }

        match self {
            Self::Value {
                gamma,
                tau,
                beta,
                sign,
            } => compute_higher_order_matrix_entry(
                source_dof,
                field_dof,
                source_index == field_index || source_elem.connectivity == field_elem.connectivity,
                source_local,
                field_local,
                source_elem,
                field_elem,
                nodes,
                physics,
                gamma,
                tau,
                beta,
                sign,
            ),
            Self::WavenumberDerivative {
                gamma,
                tau,
                beta,
                beta_derivative,
                sign,
                gauss_order,
            } => compute_higher_order_matrix_entry_wavenumber_derivative(
                source_dof,
                field_dof,
                source_index == field_index || source_elem.connectivity == field_elem.connectivity,
                source_local,
                field_local,
                source_elem,
                field_elem,
                nodes,
                physics,
                gamma,
                tau,
                beta,
                beta_derivative,
                sign,
                gauss_order,
                curved_normals,
            ),
            Self::DensityDerivative {
                gamma,
                tau,
                beta,
                sign,
            } => compute_higher_order_matrix_entry_density_derivative(
                source_dof,
                field_dof,
                source_index == field_index || source_elem.connectivity == field_elem.connectivity,
                source_local,
                field_local,
                source_elem,
                field_elem,
                nodes,
                physics,
                gamma,
                tau,
                beta,
                sign,
            ),
        }
    }
}

// ============================================================================
// H-matrix structures
// ============================================================================

/// Storage precision for the low-rank `(U, V)` factors of an `HMatrix`
/// far-field block.
///
/// `F64` is the default and matches the build-time precision of ACA — the
/// matvec then runs end-to-end in `Complex64` and the H-matrix reconstructs
/// the dense block to round-off (≲ 1×10⁻¹⁵).
///
/// `F32` halves far-field memory by storing `(U, V)` as `Complex32`. The
/// matvec upcasts to `Complex64` on the fly with an `f64` accumulator, so
/// only the *block-relative* error is `f32`-bounded (≈ 1×10⁻⁷); the global
/// matvec residual depends on the largest single block contribution and
/// typically lands in the `1×10⁻⁶`–`1×10⁻⁵` band. Suitable for the outer
/// loop of GMRES-IR (Phase 1.6) and large-N memory-constrained solves; not
/// suitable for the strict round-off parity gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LowRankPrecision {
    /// Store `(U, V)` as `Complex64`. Matches ACA build precision; matvec
    /// reconstructs dense to round-off.
    #[default]
    F64,
    /// Store `(U, V)` as `Complex32`, halving far-field memory. Matvec
    /// upcasts to `Complex64`/`f64` accumulator. Block-relative error
    /// floor ≈ 1×10⁻⁷.
    F32,
}

/// Precision-tagged storage for the `(U, V)` factors of a low-rank block.
///
/// Variants share a common shape contract: `U` is `(n_src × rank)`, `V` is
/// `(n_fld × rank)`, and the block reconstructs as
/// `A[i, j] = Σ_l U[i, l] · V[j, l]` (no conjugation). Use the
/// `apply` / `apply_transpose` helpers when consuming — both upcast to
/// `Complex64` regardless of storage precision so the calling matvec stays
/// in `Complex64`.
#[derive(Clone)]
pub enum LowRankFactors {
    /// Double-precision storage. Matches ACA build precision exactly;
    /// matvec recovers the dense block to round-off.
    F64 {
        /// `(n_src × rank)` factor.
        u: DenseMatrix<Complex64>,
        /// `(n_fld × rank)` factor.
        v: DenseMatrix<Complex64>,
    },
    /// Single-precision storage. Halves far-field memory at the cost of a
    /// `≈1×10⁻⁷` block-relative error floor; matvec upcasts to
    /// `Complex64` with an `f64` accumulator.
    ///
    /// Per-block Higham-style scaling: each factor is normalised on store
    /// so its largest entry sits near `1.0` in `f32` (best precision
    /// region), and the original magnitude is held in the `f64` `scale_*`
    /// fields. Reconstruction multiplies by `scale_u * scale_v` once per
    /// output entry — single `f64` multiply outside the `f32` inner loop.
    /// This recovers the f32 precision floor across blocks with strong
    /// dynamic range (e.g. far-field `1/r` decay) where a plain downcast
    /// would lose the small entries to round-off.
    F32 {
        /// `(n_src × rank)` factor, normalised so `max|u| ≤ 1.0`.
        u: DenseMatrix<Complex32>,
        /// `(n_fld × rank)` factor, normalised so `max|v| ≤ 1.0`.
        v: DenseMatrix<Complex32>,
        /// Original `max|U|` before normalisation, in `f64`.
        scale_u: f64,
        /// Original `max|V|` before normalisation, in `f64`.
        scale_v: f64,
    },
}

/// Smallest scale factor we'll divide by when normalising for f32 storage.
/// Chosen well above `f64::MIN_POSITIVE` so the inverse stays representable
/// without subnormals; an all-zero block degenerates to scale=`MIN_SCALE`
/// and stored f32 entries of 0, which reconstructs exactly to 0.
const MIN_SCALE_F32: f64 = 1.0e-300;

impl LowRankFactors {
    /// Storage precision tag for this block.
    pub fn precision(&self) -> LowRankPrecision {
        match self {
            LowRankFactors::F64 { .. } => LowRankPrecision::F64,
            LowRankFactors::F32 { .. } => LowRankPrecision::F32,
        }
    }

    /// Realised numerical rank (number of `(u_l, v_l)` columns).
    pub fn rank(&self) -> usize {
        match self {
            LowRankFactors::F64 { u, .. } => u.ncols(),
            LowRankFactors::F32 { u, .. } => u.ncols(),
        }
    }

    /// Number of source DOFs (rows of `U`).
    pub fn n_src(&self) -> usize {
        match self {
            LowRankFactors::F64 { u, .. } => u.nrows(),
            LowRankFactors::F32 { u, .. } => u.nrows(),
        }
    }

    /// Number of field DOFs (rows of `V`).
    pub fn n_fld(&self) -> usize {
        match self {
            LowRankFactors::F64 { v, .. } => v.nrows(),
            LowRankFactors::F32 { v, .. } => v.nrows(),
        }
    }

    /// Stored byte size of the `(U, V)` factors (excluding bookkeeping).
    /// `F32` storage adds 16 bytes for the `(scale_u, scale_v)` pair —
    /// negligible against the `(m + n) · rank · 8` factor-storage cost
    /// once the block has more than a couple of entries.
    pub fn bytes(&self) -> usize {
        let entries = (self.n_src() + self.n_fld()) * self.rank();
        match self {
            LowRankFactors::F64 { .. } => entries * std::mem::size_of::<Complex64>(),
            LowRankFactors::F32 { .. } => {
                entries * std::mem::size_of::<Complex32>() + 2 * std::mem::size_of::<f64>()
            }
        }
    }

    /// Apply `U · (V^T · x)` to a `Complex64` input, upcasting `Complex32`
    /// factors on the fly through an `f64`-precision accumulator. Returns
    /// the `n_src`-dimensional result.
    ///
    /// For `F32` storage the `(scale_u · scale_v)` product is folded in
    /// once per output entry as a single `f64` multiply outside the f32
    /// inner loop, so the block reconstructs as
    /// `scale_u · scale_v · U_f32 · (V_f32^T · x_local)` with full f64
    /// magnitude even when the original block has strong dynamic range.
    pub fn apply(&self, x_local: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n_src()];
        let mut t = Vec::new();
        self.apply_into(x_local, &mut y, &mut t);
        y
    }

    /// Allocation-free variant of [`Self::apply`]: writes the `n_src`-
    /// dimensional result into `y_local`, using `t` as rank-sized scratch.
    pub fn apply_into(
        &self,
        x_local: &[Complex64],
        y_local: &mut [Complex64],
        t: &mut Vec<Complex64>,
    ) {
        match self {
            LowRankFactors::F64 { u, v } => {
                t.clear();
                t.resize(v.ncols(), Complex64::new(0.0, 0.0));
                v.matvec_transpose_into(x_local, t);
                debug_assert_eq!(y_local.len(), u.nrows());
                u.matvec_into(t, y_local);
            }
            LowRankFactors::F32 {
                u,
                v,
                scale_u,
                scale_v,
            } => {
                let n_fld = v.nrows();
                let rank = v.ncols();
                debug_assert_eq!(x_local.len(), n_fld);
                t.clear();
                t.resize(rank, Complex64::new(0.0, 0.0));
                for (l, t_l) in t.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for j in 0..n_fld {
                        let v_lj = v[[j, l]];
                        let v64 = Complex64::new(v_lj.re as f64, v_lj.im as f64);
                        acc += v64 * x_local[j];
                    }
                    *t_l = acc;
                }
                let n_src = u.nrows();
                debug_assert_eq!(y_local.len(), n_src);
                let scale = *scale_u * *scale_v;
                for (i, y_i) in y_local.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for l in 0..rank {
                        let u_il = u[[i, l]];
                        let u64 = Complex64::new(u_il.re as f64, u_il.im as f64);
                        acc += u64 * t[l];
                    }
                    *y_i = acc * scale;
                }
            }
        }
    }

    /// Apply a low-rank block to multiple right-hand sides at once:
    /// `Y = U · (Vᵀ · X)`.
    ///
    /// The `F64` path uses dense matrix-matrix products so the BLAS-backed
    /// `Complex64` implementation can reuse the rank projection across all
    /// right-hand sides. `F32` keeps the same algebra and scaling contract,
    /// with an explicit f64 accumulation during the upcast.
    pub fn apply_block_into(
        &self,
        x_local: &DenseMatrix<Complex64>,
        y_local: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(x_local.nrows(), self.n_fld());
        assert_eq!(y_local.nrows(), self.n_src());
        assert_eq!(x_local.ncols(), y_local.ncols());

        match self {
            LowRankFactors::F64 { u, v } => {
                let v_transposed = v.transposed();
                let mut projected = DenseMatrix::<Complex64>::zeros(v.ncols(), x_local.ncols());
                v_transposed.matmul_into(x_local, &mut projected);
                u.matmul_into(&projected, y_local);
            }
            LowRankFactors::F32 {
                u,
                v,
                scale_u,
                scale_v,
            } => {
                y_local.fill(Complex64::new(0.0, 0.0));
                let mut projected = DenseMatrix::<Complex64>::zeros(v.ncols(), x_local.ncols());
                for l in 0..v.ncols() {
                    for rhs in 0..x_local.ncols() {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for j in 0..v.nrows() {
                            let v_lj = v[[j, l]];
                            acc +=
                                Complex64::new(v_lj.re as f64, v_lj.im as f64) * x_local[[j, rhs]];
                        }
                        projected[[l, rhs]] = acc;
                    }
                }

                let scale = *scale_u * *scale_v;
                for i in 0..u.nrows() {
                    for rhs in 0..x_local.ncols() {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for l in 0..u.ncols() {
                            let u_il = u[[i, l]];
                            acc += Complex64::new(u_il.re as f64, u_il.im as f64)
                                * projected[[l, rhs]];
                        }
                        y_local[[i, rhs]] = acc * scale;
                    }
                }
            }
        }
    }

    /// Apply the algebraic-transpose low-rank block to several right-hand
    /// sides at once: `Y = V · (U^T · X)`.
    ///
    /// Keeping this operation block-aware avoids rebuilding the rank
    /// projection for every column when an H-matrix transpose action is
    /// used by block Krylov or adjoint solves.
    pub fn apply_transpose_block_into(
        &self,
        x_local: &DenseMatrix<Complex64>,
        y_local: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(x_local.nrows(), self.n_src());
        assert_eq!(y_local.nrows(), self.n_fld());
        assert_eq!(x_local.ncols(), y_local.ncols());

        match self {
            LowRankFactors::F64 { u, v } => {
                let u_transposed = u.transposed();
                let mut projected = DenseMatrix::<Complex64>::zeros(u.ncols(), x_local.ncols());
                u_transposed.matmul_into(x_local, &mut projected);
                v.matmul_into(&projected, y_local);
            }
            LowRankFactors::F32 {
                u,
                v,
                scale_u,
                scale_v,
            } => {
                let rank = u.ncols();
                let nrhs = x_local.ncols();
                let mut projected = DenseMatrix::<Complex64>::zeros(rank, nrhs);
                for l in 0..rank {
                    for rhs in 0..nrhs {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for i in 0..u.nrows() {
                            let u_il = u[[i, l]];
                            acc +=
                                Complex64::new(u_il.re as f64, u_il.im as f64) * x_local[[i, rhs]];
                        }
                        projected[[l, rhs]] = acc;
                    }
                }

                let scale = *scale_u * *scale_v;
                for j in 0..v.nrows() {
                    for rhs in 0..nrhs {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for l in 0..rank {
                            let v_jl = v[[j, l]];
                            acc += Complex64::new(v_jl.re as f64, v_jl.im as f64)
                                * projected[[l, rhs]];
                        }
                        y_local[[j, rhs]] = acc * scale;
                    }
                }
            }
        }
    }

    /// Apply `V · (U^T · x)` (the algebraic transpose of `apply`) to a
    /// `Complex64` input, upcasting on the fly. Returns the
    /// `n_fld`-dimensional result. Same `(scale_u · scale_v)` fold-in as
    /// `apply` for the `F32` storage variant.
    pub fn apply_transpose(&self, x_local: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n_fld()];
        let mut t = Vec::new();
        self.apply_transpose_into(x_local, &mut y, &mut t);
        y
    }

    /// Allocation-free variant of [`Self::apply_transpose`]: writes the
    /// `n_fld`-dimensional result into `y_local`, using `t` as rank-sized
    /// scratch.
    pub fn apply_transpose_into(
        &self,
        x_local: &[Complex64],
        y_local: &mut [Complex64],
        t: &mut Vec<Complex64>,
    ) {
        match self {
            LowRankFactors::F64 { u, v } => {
                t.clear();
                t.resize(u.ncols(), Complex64::new(0.0, 0.0));
                u.matvec_transpose_into(x_local, t);
                debug_assert_eq!(y_local.len(), v.nrows());
                v.matvec_into(t, y_local);
            }
            LowRankFactors::F32 {
                u,
                v,
                scale_u,
                scale_v,
            } => {
                let n_src = u.nrows();
                let rank = u.ncols();
                debug_assert_eq!(x_local.len(), n_src);
                t.clear();
                t.resize(rank, Complex64::new(0.0, 0.0));
                for (l, t_l) in t.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for i in 0..n_src {
                        let u_il = u[[i, l]];
                        let u64 = Complex64::new(u_il.re as f64, u_il.im as f64);
                        acc += u64 * x_local[i];
                    }
                    *t_l = acc;
                }
                let n_fld = v.nrows();
                debug_assert_eq!(y_local.len(), n_fld);
                let scale = *scale_u * *scale_v;
                for (j, y_j) in y_local.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for l in 0..rank {
                        let v_jl = v[[j, l]];
                        let v64 = Complex64::new(v_jl.re as f64, v_jl.im as f64);
                        acc += v64 * t[l];
                    }
                    *y_j = acc * scale;
                }
            }
        }
    }

    /// Apply `conj(V) · (conj(U)^T · x)` — the Hermitian (conjugate
    /// transpose) of the block `A[i, j] = Σ_l U[i, l] · V[j, l]`:
    /// `y_j = Σ_i conj(A[i, j]) · x_i = Σ_l conj(V[j, l]) · (Σ_i conj(U[i, l]) · x_i)`.
    ///
    /// Mind the storage convention: the factors reconstruct with `V^T`,
    /// **not** `V^H`, so the Hermitian apply conjugates *both* factors —
    /// it is `conj(A^T · conj(x))`, not `V · (U^H · x)`.
    pub fn apply_hermitian(&self, x_local: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n_fld()];
        let mut t = Vec::new();
        self.apply_hermitian_into(x_local, &mut y, &mut t);
        y
    }

    /// Allocation-free variant of [`Self::apply_hermitian`]: writes the
    /// `n_fld`-dimensional result into `y_local`, using `t` as rank-sized
    /// scratch.
    pub fn apply_hermitian_into(
        &self,
        x_local: &[Complex64],
        y_local: &mut [Complex64],
        t: &mut Vec<Complex64>,
    ) {
        match self {
            LowRankFactors::F64 { u, v } => {
                // t = conj(U)^T · x via the ConjTrans GEMV kernel.
                t.clear();
                t.resize(u.ncols(), Complex64::new(0.0, 0.0));
                u.matvec_hermitian_into(x_local, t);
                // y = conj(V) · t = conj(V · conj(t)): conj the rank-sized
                // scratch, run the plain GEMV, conj the output.
                for t_l in t.iter_mut() {
                    *t_l = t_l.conj();
                }
                debug_assert_eq!(y_local.len(), v.nrows());
                v.matvec_into(t, y_local);
                for y_j in y_local.iter_mut() {
                    *y_j = y_j.conj();
                }
            }
            LowRankFactors::F32 {
                u,
                v,
                scale_u,
                scale_v,
            } => {
                let n_src = u.nrows();
                let rank = u.ncols();
                debug_assert_eq!(x_local.len(), n_src);
                t.clear();
                t.resize(rank, Complex64::new(0.0, 0.0));
                for (l, t_l) in t.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for i in 0..n_src {
                        let u_il = u[[i, l]];
                        let u64 = Complex64::new(u_il.re as f64, u_il.im as f64);
                        acc += u64.conj() * x_local[i];
                    }
                    *t_l = acc;
                }
                let n_fld = v.nrows();
                debug_assert_eq!(y_local.len(), n_fld);
                // `scale_u`/`scale_v` are real f64 magnitudes — unchanged by
                // conjugation.
                let scale = *scale_u * *scale_v;
                for (j, y_j) in y_local.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for l in 0..rank {
                        let v_jl = v[[j, l]];
                        let v64 = Complex64::new(v_jl.re as f64, v_jl.im as f64);
                        acc += v64.conj() * t[l];
                    }
                    *y_j = acc * scale;
                }
            }
        }
    }
}

/// Build a Higham-scaled `F32` low-rank block from `f64` factors. Each
/// factor is normalised by its own `max|·|` so the stored `f32` entries
/// sit near `1.0` (best precision region), and the original magnitude is
/// held in the `scale_*` fields. Reconstruction (see
/// `LowRankFactors::apply` / `apply_transpose`) multiplies by
/// `scale_u * scale_v` once per output entry as a single `f64` multiply
/// outside the f32 inner loop.
///
/// All-zero blocks: `MIN_SCALE_F32` floors the scale so the inverse stays
/// finite; the stored f32 entries are 0 either way and the reconstructed
/// block is exactly 0.
fn downcast_factors_f64_to_f32(
    u: DenseMatrix<Complex64>,
    v: DenseMatrix<Complex64>,
) -> LowRankFactors {
    let scale_u = u
        .as_slice()
        .iter()
        .map(|c| c.norm())
        .fold(0.0_f64, f64::max)
        .max(MIN_SCALE_F32);
    let scale_v = v
        .as_slice()
        .iter()
        .map(|c| c.norm())
        .fold(0.0_f64, f64::max)
        .max(MIN_SCALE_F32);
    let inv_u = 1.0 / scale_u;
    let inv_v = 1.0 / scale_v;
    let (u_rows, u_cols) = u.shape();
    let (v_rows, v_cols) = v.shape();
    let u32_data: Vec<Complex32> = u
        .as_slice()
        .iter()
        .map(|c| Complex32::new((c.re * inv_u) as f32, (c.im * inv_u) as f32))
        .collect();
    let v32_data: Vec<Complex32> = v
        .as_slice()
        .iter()
        .map(|c| Complex32::new((c.re * inv_v) as f32, (c.im * inv_v) as f32))
        .collect();
    let u32 = DenseMatrix::from_row_major_vec(u_rows, u_cols, u32_data);
    let v32 = DenseMatrix::from_row_major_vec(v_rows, v_cols, v32_data);
    LowRankFactors::F32 {
        u: u32,
        v: v32,
        scale_u,
        scale_v,
    }
}

/// A low-rank block `U V^T` approximation of a far-field interaction.
///
/// Storage convention: entries reconstruct as
/// `A[i, j] = Σ_l U[i, l] · V[j, l]` (no Hermitian conjugation). This matches
/// the residual update inside the ACA loops and the `matvec` / `apply_transpose`
/// implementations on `HMatrix`. See `test_hmatrix_matvec_dense_parity_complex`.
///
/// Storage precision is selected by the builder via `LowRankPrecision`. The
/// matvec helpers on `LowRankFactors` upcast `Complex32` factors back to
/// `Complex64` on the fly so callers see a uniform `Complex64` interface
/// regardless of how the factors are stored.
#[derive(Clone)]
pub struct LowRankBlock {
    /// Source cluster index
    pub source_cluster: usize,
    /// Field cluster index
    pub field_cluster: usize,
    /// Stored `(U, V)` factors — see `LowRankFactors`.
    pub factors: LowRankFactors,
    /// Numerical rank of the approximation
    pub rank: usize,
    /// Effective maximum rank that ACA was permitted for this block
    /// (i.e. `min(user_max_rank, m, n)` after cluster-size clipping).
    pub requested_max_rank: usize,
    /// True iff ACA hit its tolerance criterion or exhausted the block exactly.
    /// False means the rank cap was binding — the approximation may be under-resolved.
    pub converged: bool,
}

/// Near-field dense block
#[derive(Clone)]
pub struct DenseBlock {
    /// Source cluster index
    pub source_cluster: usize,
    /// Field cluster index
    pub field_cluster: usize,
    /// Dense coefficient matrix for this block
    pub matrix: DenseMatrix<Complex64>,
}

/// NUMA-friendly row-cluster partition of an H-matrix.
///
/// Partitions are contiguous in Morton order over source-cluster centres.
/// Each partition owns a contiguous range in `dense_blocks` and
/// `low_rank_blocks`, so later matvecs can process spatially local row blocks
/// independently. This mirrors the CSR row-partition strategy in
/// `sonium-solvers`: split by output rows, allocate block storage per
/// partition during build, and avoid cross-partition writes during matvec.
#[derive(Debug, Clone)]
pub struct HMatrixPartition {
    /// Source cluster indices in this partition, ordered by Morton key.
    pub cluster_indices: Vec<usize>,
    /// First dense block belonging to this partition.
    pub dense_start: usize,
    /// One-past-last dense block belonging to this partition.
    pub dense_end: usize,
    /// First low-rank block belonging to this partition.
    pub low_rank_start: usize,
    /// One-past-last low-rank block belonging to this partition.
    pub low_rank_end: usize,
    /// Number of output DOFs covered by the source clusters in this partition.
    pub num_dofs: usize,
}

/// NUMA-friendly field-cluster (column) partition of an H-matrix, used to
/// schedule transpose and Hermitian matvecs.
///
/// Mirrors [`HMatrixPartition`] but keyed by **field** cluster: transpose
/// output DOFs live on the field side, so partitioning by field cluster
/// keeps partition writes disjoint exactly like the row schedule does for
/// the forward matvec. Because `dense_blocks` / `low_rank_blocks` are
/// stored in source-cluster (row) order, the per-partition block ranges
/// index into the `transpose_dense_order` / `transpose_low_rank_order`
/// indirection arrays instead of the block arrays directly.
#[derive(Debug, Clone)]
pub struct HMatrixTransposePartition {
    /// Field cluster indices in this partition, ordered by Morton key.
    pub cluster_indices: Vec<usize>,
    /// First entry in `transpose_dense_order` belonging to this partition.
    pub dense_start: usize,
    /// One-past-last entry in `transpose_dense_order` for this partition.
    pub dense_end: usize,
    /// First entry in `transpose_low_rank_order` belonging to this partition.
    pub low_rank_start: usize,
    /// One-past-last entry in `transpose_low_rank_order` for this partition.
    pub low_rank_end: usize,
    /// Number of output DOFs covered by the field clusters in this partition.
    pub num_dofs: usize,
}

/// H-matrix: hierarchical decomposition with near-field dense + far-field low-rank
#[derive(Clone)]
pub struct HMatrix {
    /// Total number of DOFs
    pub num_dofs: usize,
    /// Dense near-field blocks
    pub dense_blocks: Vec<DenseBlock>,
    /// Low-rank far-field blocks
    pub low_rank_blocks: Vec<LowRankBlock>,
    /// Cluster DOF mappings: for each cluster, the global DOF indices
    pub cluster_dof_indices: Vec<Vec<usize>>,
    /// Spatial row-cluster partitions used for NUMA-aware matvec scheduling.
    pub partitions: Vec<HMatrixPartition>,
    /// `dense_blocks` indices sorted by field cluster in Morton order; the
    /// indirection backing [`Self::transpose_partitions`].
    pub transpose_dense_order: Vec<usize>,
    /// `low_rank_blocks` indices sorted by field cluster in Morton order.
    pub transpose_low_rank_order: Vec<usize>,
    /// Spatial field-cluster partitions used for NUMA-aware
    /// transpose/Hermitian matvec scheduling.
    pub transpose_partitions: Vec<HMatrixTransposePartition>,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Compression ratio vs dense (dense_bytes / hmatrix_bytes)
    pub compression_ratio: f64,
}

/// Per-block rank statistics for a populated H-matrix.
///
/// `saturated_count` is the number of low-rank blocks where ACA terminated
/// because the user-supplied `max_rank` cap was reached rather than the
/// tolerance criterion. A non-zero saturation fraction means the H-matrix
/// is silently under-resolved and either `max_rank` should be raised or the
/// approximation scheme upgraded (HCA/ACA-GP — see Stage 1.1 in `bem-step1.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankStats {
    /// Number of low-rank blocks contributing to the statistics.
    pub block_count: usize,
    /// Smallest realised rank across all low-rank blocks.
    pub min_rank: usize,
    /// Largest realised rank across all low-rank blocks.
    pub max_rank: usize,
    /// Mean realised rank.
    pub mean_rank: f64,
    /// 50th percentile realised rank.
    pub p50_rank: usize,
    /// 90th percentile realised rank.
    pub p90_rank: usize,
    /// 99th percentile realised rank.
    pub p99_rank: usize,
    /// Mean of the per-block effective rank caps that ACA was permitted.
    pub mean_requested_max_rank: f64,
    /// Number of blocks where ACA hit the user-supplied rank cap before
    /// reaching the tolerance criterion (i.e. silent under-resolution).
    pub saturated_count: usize,
    /// `saturated_count / block_count` (zero when there are no low-rank blocks).
    pub saturation_fraction: f64,
}

impl std::fmt::Display for RankStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RankStats {{ blocks: {}, rank min/p50/mean/p90/p99/max: {}/{}/{:.2}/{}/{}/{}, \
             mean_cap: {:.2}, saturated: {} ({:.2}%) }}",
            self.block_count,
            self.min_rank,
            self.p50_rank,
            self.mean_rank,
            self.p90_rank,
            self.p99_rank,
            self.max_rank,
            self.mean_requested_max_rank,
            self.saturated_count,
            self.saturation_fraction * 100.0,
        )
    }
}

// ============================================================================
// Matrix entry computation
// ============================================================================

// Internal element proxies let the existing ACA kernels operate on one row
// per higher-order DOF without duplicating the pivot/termination logic.  The
// high bit is reserved for the tag; the next byte stores the panel basis size
// and the low byte stores the local basis index.  These values are bounded by
// the supported Tri3/Quad4 linear and quadratic layouts.
const DOF_PROXY_TAG: usize = 1usize << (usize::BITS - 1);
/// Marker used by the continuous/shared-DOF incidence expansion.  Expanded
/// proxies already represent one global DOF incidence, so the builder must
/// not run its ordinary one-proxy-per-global-DOF deduplication again.
pub(crate) const SHARED_DOF_PROXY_TAG: usize = 1usize << (usize::BITS - 2);
const DOF_PROXY_BASIS_SHIFT: usize = 8;
const DOF_PROXY_BASIS_MASK: usize = 0xffusize << DOF_PROXY_BASIS_SHIFT;
const DOF_PROXY_LOCAL_MASK: usize = 0xff;

fn encode_dof_proxy_group(num_basis: usize, local: usize) -> usize {
    DOF_PROXY_TAG | ((num_basis & 0xff) << DOF_PROXY_BASIS_SHIFT) | (local & DOF_PROXY_LOCAL_MASK)
}

pub(crate) fn encode_shared_dof_proxy_group(num_basis: usize, local: usize) -> usize {
    SHARED_DOF_PROXY_TAG | encode_dof_proxy_group(num_basis, local)
}

fn has_shared_dof_proxy_group(group: usize) -> bool {
    group & SHARED_DOF_PROXY_TAG != 0
}

fn decode_dof_proxy_group(group: usize) -> Option<(usize, usize)> {
    if group & DOF_PROXY_TAG == 0 {
        return None;
    }
    let num_basis = (group & DOF_PROXY_BASIS_MASK) >> DOF_PROXY_BASIS_SHIFT;
    let local = group & DOF_PROXY_LOCAL_MASK;
    (num_basis > 1).then_some((num_basis, local))
}

fn basis_order_for_dof_count(element: &Element, num_basis: usize) -> Option<BasisOrder> {
    match (element.element_type, num_basis) {
        (crate::core::types::ElementType::Tri3, 3)
        | (crate::core::types::ElementType::Quad4, 4) => Some(BasisOrder::Linear),
        (crate::core::types::ElementType::Tri3, 6)
        | (crate::core::types::ElementType::Quad4, 9) => Some(BasisOrder::Quadratic),
        _ => None,
    }
}

fn shared_proxy_curved_normals(source_elem: &Element, field_elem: &Element) -> bool {
    let shared_proxy = has_shared_dof_proxy_group(source_elem.group)
        || has_shared_dof_proxy_group(field_elem.group);
    shared_proxy
        && source_elem
            .node_normals
            .iter()
            .chain(field_elem.node_normals.iter())
            .any(|normal| normal.iter().any(|value| value.abs() > 1.0e-15))
}

#[allow(clippy::too_many_arguments)]
fn compute_higher_order_matrix_entry(
    source_dof: usize,
    field_dof: usize,
    self_interaction: bool,
    source_local: usize,
    field_local: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    sign: f64,
) -> Complex64 {
    // Continuous higher-order H-matrix assembly uses shared-incidence proxy
    // elements and must mirror the curved field quadrature used by the dense
    // continuous TBEM reference.  The ordinary discontinuous proxy path keeps
    // its historical affine kernel, whose independent exact-TBEM certificate
    // is intentionally defined with flat-panel quadrature.
    let curved_normals = shared_proxy_curved_normals(source_elem, field_elem);
    compute_higher_order_matrix_entry_with_normals(
        source_dof,
        field_dof,
        self_interaction,
        source_local,
        field_local,
        source_elem,
        field_elem,
        nodes,
        physics,
        gamma,
        tau,
        beta,
        sign,
        curved_normals,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_higher_order_matrix_entry_with_normals(
    source_dof: usize,
    field_dof: usize,
    self_interaction: bool,
    source_local: usize,
    field_local: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    sign: f64,
    curved_normals: bool,
) -> Complex64 {
    let source_basis_count = decode_dof_proxy_group(source_elem.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| source_elem.dof_addresses.len());
    let field_basis_count = decode_dof_proxy_group(field_elem.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| field_elem.dof_addresses.len());
    let Some(basis) = basis_order_for_dof_count(source_elem, source_basis_count)
        .or_else(|| basis_order_for_dof_count(field_elem, field_basis_count))
    else {
        // The public admission layer rejects unsupported layouts. Keep this
        // helper fail-closed if a malformed internal proxy reaches it.
        return Complex64::new(f64::NAN, f64::NAN);
    };
    let source_points = source_elem.collocation_points(basis, nodes);
    let source_point = source_points
        .get(source_local)
        .copied()
        .unwrap_or(source_elem.center);
    let source_normal = if curved_normals {
        source_elem
            .collocation_normals(basis, nodes)
            .get(source_local)
            .copied()
            .unwrap_or(source_elem.normal)
    } else {
        source_elem.normal
    };
    let element_coords = get_element_coords(field_elem, nodes);
    let field_bc_type = get_bc_type_and_value(&field_elem.boundary_condition).0;
    let curved = curved_normals
        && field_elem
            .node_normals
            .iter()
            .any(|normal| normal.iter().any(|value| value.abs() > 1.0e-15));
    let field_result = if self_interaction {
        if curved {
            singular_integration_linear_curved(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                physics,
                field_basis_count,
                Some(field_elem.node_normals.as_slice()),
            )
        } else {
            singular_integration_linear(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                physics,
                field_basis_count,
            )
        }
    } else {
        if curved {
            regular_integration_linear_with_normals(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                field_elem.area,
                physics,
                field_basis_count,
                Some(field_elem.node_normals.as_slice()),
            )
        } else {
            regular_integration_linear(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                field_elem.area,
                physics,
                field_basis_count,
            )
        }
    };
    let Some(result) = field_result.basis_results.get(field_local) else {
        return Complex64::new(f64::NAN, f64::NAN);
    };

    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
    let mut coefficient = match field_bc_type {
        0 => sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta),
        1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
        _ => Complex64::new(0.0, 0.0),
    };
    if field_bc_type == 0
        && let Some(admittance) = field_elem.boundary_condition.admittance()
    {
        coefficient -= sign
            * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
            * admittance
            * i_omega_rho;
    }

    // The free Burton-Miller trace term is diagonal in the collocation DOF
    // layout. Add it here for near-field blocks so H-matrix/TBEM parity holds
    // for every local basis slot, not only the first panel DOF.
    if self_interaction && source_dof == field_dof {
        match field_bc_type {
            0 => {
                coefficient -= sign * gamma * 0.5;
                if let Some(admittance) = field_elem.boundary_condition.admittance() {
                    coefficient -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                }
            }
            1 => {
                coefficient += sign * beta * tau * 0.5;
            }
            _ => {}
        }
    }
    coefficient
}

/// Compute a single BEM matrix entry A[i, j] using the Burton-Miller formulation.
///
/// This evaluates the kernel between source element i and field element j,
/// using `regular_integration()` for off-diagonal entries. For self-interaction
/// (i == j), the result includes singular integration effects, but ACA only
/// evaluates far-field blocks where i != j by construction.
/// Compute a basis-weighted higher-order matrix entry differentiated
/// analytically with respect to acoustic wavenumber.
///
/// The value and derivative use the same local basis layout.  Self blocks use
/// the singular Duffy/edge derivative path; separated blocks use the fixed
/// far-field quadrature derivative.  Burton--Miller beta and admittance
/// pressure-factor terms are differentiated explicitly, including the local
/// jump contribution.
#[allow(clippy::too_many_arguments)]
fn compute_higher_order_matrix_entry_wavenumber_derivative(
    source_dof: usize,
    field_dof: usize,
    self_interaction: bool,
    source_local: usize,
    field_local: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    gauss_order: usize,
    curved_normals: bool,
) -> Complex64 {
    let source_basis_count = decode_dof_proxy_group(source_elem.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| source_elem.dof_addresses.len());
    let field_basis_count = decode_dof_proxy_group(field_elem.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| field_elem.dof_addresses.len());
    let Some(basis) = basis_order_for_dof_count(source_elem, source_basis_count)
        .or_else(|| basis_order_for_dof_count(field_elem, field_basis_count))
    else {
        return Complex64::new(f64::NAN, f64::NAN);
    };

    let source_point = source_elem
        .collocation_points(basis, nodes)
        .get(source_local)
        .copied()
        .unwrap_or(source_elem.center);
    let source_normal = source_elem.normal;
    let element_coords = get_element_coords(field_elem, nodes);
    let field_bc_type = get_bc_type_and_value(&field_elem.boundary_condition).0;
    let curved = field_elem
        .node_normals
        .iter()
        .any(|normal| normal.iter().any(|value| value.abs() > 1.0e-15));
    let value_result = if self_interaction {
        if curved_normals && curved {
            singular_integration_linear_curved(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                physics,
                field_basis_count,
                Some(field_elem.node_normals.as_slice()),
            )
        } else {
            singular_integration_linear(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                physics,
                field_basis_count,
            )
        }
    } else if curved_normals && curved {
        crate::core::integration::regular_integration_linear_with_normals(
            &source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            field_basis_count,
            Some(field_elem.node_normals.as_slice()),
        )
    } else {
        regular_integration_linear(
            &source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            field_basis_count,
        )
    };
    let derivative_result = if self_interaction {
        if curved_normals && curved {
            singular_integration_linear_curved_wavenumber_derivative(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                physics,
                field_basis_count,
                Some(field_elem.node_normals.as_slice()),
            )
        } else {
            singular_integration_linear_wavenumber_derivative(
                &source_point,
                &source_normal,
                &element_coords,
                field_elem.element_type,
                physics,
                field_basis_count,
            )
        }
    } else if curved_normals && curved {
        regular_integration_linear_fixed_order_wavenumber_derivative_with_normals(
            &source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            physics,
            field_basis_count,
            gauss_order,
            Some(field_elem.node_normals.as_slice()),
        )
    } else {
        regular_integration_linear_fixed_order_wavenumber_derivative(
            &source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            physics,
            field_basis_count,
            gauss_order,
        )
    };
    let Some(value) = value_result.basis_results.get(field_local) else {
        return Complex64::new(f64::NAN, f64::NAN);
    };
    let Some(derivative) = derivative_result.basis_results.get(field_local) else {
        return Complex64::new(f64::NAN, f64::NAN);
    };

    let pressure_factor = Complex64::new(0.0, physics.pressure_factor);
    let derivative_pressure_factor =
        Complex64::new(0.0, physics.pressure_factor_wavenumber_derivative());
    let mut coefficient = match field_bc_type {
        0 => {
            sign * (derivative.dg_dn_integral * gamma * tau
                + derivative.d2g_dnxdny_integral * beta
                + value.d2g_dnxdny_integral * beta_derivative)
        }
        1 => {
            -sign
                * (derivative.g_integral * gamma * tau
                    + derivative.dg_dnx_integral * beta
                    + value.dg_dnx_integral * beta_derivative)
        }
        _ => Complex64::new(0.0, 0.0),
    };

    if field_bc_type == 0
        && let Some(admittance) = field_elem.boundary_condition.admittance()
    {
        coefficient -= sign
            * (derivative.g_integral * gamma * tau
                + derivative.dg_dnx_integral * beta
                + value.dg_dnx_integral * beta_derivative)
            * admittance
            * pressure_factor;
        coefficient -= sign
            * (value.g_integral * gamma * tau + value.dg_dnx_integral * beta)
            * admittance
            * derivative_pressure_factor;
    }

    if self_interaction && source_dof == field_dof {
        match field_bc_type {
            0 => {
                coefficient -= sign
                    * admittance_jump_term(
                        field_elem,
                        tau,
                        beta,
                        beta_derivative,
                        pressure_factor,
                        derivative_pressure_factor,
                    );
            }
            1 => {
                coefficient += sign * beta_derivative * tau * 0.5;
            }
            _ => {}
        }
    }

    coefficient
}

fn admittance_jump_term(
    field_elem: &Element,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    pressure_factor: Complex64,
    derivative_pressure_factor: Complex64,
) -> Complex64 {
    let Some(admittance) = field_elem.boundary_condition.admittance() else {
        return Complex64::new(0.0, 0.0);
    };
    admittance * tau * 0.5 * (derivative_pressure_factor * beta + pressure_factor * beta_derivative)
}

/// Compute the fixed-geometry density derivative of one constant-panel entry.
///
/// The Helmholtz kernels and Burton--Miller coefficients are density
/// independent.  Only impedance traces contain `i * omega * rho`, so the
/// derivative is the impedance contribution with `rho` replaced by one.
#[allow(clippy::too_many_arguments)]
fn compute_matrix_entry_density_derivative(
    src_idx: usize,
    fld_idx: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    sign: f64,
) -> Complex64 {
    if let (Some((_, source_local)), Some((_, field_local))) = (
        decode_dof_proxy_group(source_elem.group),
        decode_dof_proxy_group(field_elem.group),
    ) {
        return compute_higher_order_matrix_entry_density_derivative(
            source_elem.dof_addresses[0],
            field_elem.dof_addresses[0],
            src_idx == fld_idx || source_elem.connectivity == field_elem.connectivity,
            source_local,
            field_local,
            source_elem,
            field_elem,
            nodes,
            physics,
            gamma,
            tau,
            beta,
            sign,
        );
    }
    let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
    let Some(admittance) = field_elem.boundary_condition.admittance() else {
        return Complex64::new(0.0, 0.0);
    };
    if field_bc_type != 0 {
        return Complex64::new(0.0, 0.0);
    }

    let element_coords = get_element_coords(field_elem, nodes);
    let source_point = &source_elem.center;
    let source_normal = source_elem.normal;
    let self_interaction = src_idx == fld_idx;
    let result = if self_interaction {
        singular_integration_curved(
            source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            physics,
            None,
            field_bc_type,
            false,
            None,
        )
    } else {
        regular_integration_with_normals(
            source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            None,
            field_bc_type,
            false,
            None,
        )
    };
    let i_omega_per_density = Complex64::new(0.0, physics.omega * physics.harmonic_factor);
    let pressure_trace = result.g_integral * gamma * tau + result.dg_dnx_integral * beta;
    let mut derivative = -sign * pressure_trace * admittance * i_omega_per_density;
    if self_interaction {
        derivative -= sign * admittance * i_omega_per_density * beta * tau * 0.5;
    }
    derivative
}

/// Higher-order counterpart of [`compute_matrix_entry_density_derivative`].
/// It mirrors the value entry's basis-weighted quadrature and diagonal jump
/// term, while differentiating only the explicit impedance pressure factor.
#[allow(clippy::too_many_arguments)]
fn compute_higher_order_matrix_entry_density_derivative(
    source_dof: usize,
    field_dof: usize,
    self_interaction: bool,
    source_local: usize,
    field_local: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    sign: f64,
) -> Complex64 {
    let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
    let Some(admittance) = field_elem.boundary_condition.admittance() else {
        return Complex64::new(0.0, 0.0);
    };
    if field_bc_type != 0 {
        return Complex64::new(0.0, 0.0);
    }
    let source_basis_count = decode_dof_proxy_group(source_elem.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| source_elem.dof_addresses.len());
    let field_basis_count = decode_dof_proxy_group(field_elem.group)
        .map(|(count, _)| count)
        .unwrap_or_else(|| field_elem.dof_addresses.len());
    let Some(basis) = basis_order_for_dof_count(source_elem, source_basis_count)
        .or_else(|| basis_order_for_dof_count(field_elem, field_basis_count))
    else {
        return Complex64::new(f64::NAN, f64::NAN);
    };
    let source_point = source_elem
        .collocation_points(basis, nodes)
        .get(source_local)
        .copied()
        .unwrap_or(source_elem.center);
    let source_normal = source_elem.normal;
    let element_coords = get_element_coords(field_elem, nodes);
    let result = if self_interaction {
        singular_integration_linear(
            &source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            physics,
            field_basis_count,
        )
    } else {
        regular_integration_linear(
            &source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            field_basis_count,
        )
    };
    let Some(result) = result.basis_results.get(field_local) else {
        return Complex64::new(f64::NAN, f64::NAN);
    };
    let i_omega_per_density = Complex64::new(0.0, physics.omega * physics.harmonic_factor);
    let pressure_trace = result.g_integral * gamma * tau + result.dg_dnx_integral * beta;
    let mut derivative = -sign * pressure_trace * admittance * i_omega_per_density;
    if self_interaction && source_dof == field_dof {
        derivative -= sign * admittance * i_omega_per_density * beta * tau * 0.5;
    }
    derivative
}

fn compute_matrix_entry(
    src_idx: usize,
    fld_idx: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    sign: f64,
) -> Complex64 {
    if let (Some((_, source_local)), Some((_, field_local))) = (
        decode_dof_proxy_group(source_elem.group),
        decode_dof_proxy_group(field_elem.group),
    ) {
        return compute_higher_order_matrix_entry(
            source_elem.dof_addresses[0],
            field_elem.dof_addresses[0],
            src_idx == fld_idx || source_elem.connectivity == field_elem.connectivity,
            source_local,
            field_local,
            source_elem,
            field_elem,
            nodes,
            physics,
            gamma,
            tau,
            beta,
            sign,
        );
    }

    use std::cell::RefCell;

    thread_local! {
        static COORDS_BUF: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(24));
    }

    let source_point = &source_elem.center;
    let source_normal = source_elem.normal;

    // Get field element coordinates — reuse a thread-local buffer to avoid
    // allocating a DenseMatrix on every call (hot path in ACA loops).
    let num_nodes = field_elem.connectivity.len();
    let element_coords = COORDS_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        for &node_idx in field_elem.connectivity.iter() {
            for &coord in nodes[node_idx].iter().take(3) {
                buf.push(coord);
            }
        }
        let taken = std::mem::take(&mut *buf);
        DenseMatrix::from_row_major_vec(num_nodes, 3, taken)
    });

    let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
    let node_normals = None;

    let self_interaction = src_idx == fld_idx;

    let result = if self_interaction {
        singular_integration_curved(
            source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            physics,
            None,
            field_bc_type,
            false,
            node_normals,
        )
    } else {
        regular_integration_with_normals(
            source_point,
            &source_normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            None,
            field_bc_type,
            false,
            node_normals,
        )
    };

    let mut coeff = match field_bc_type {
        0 => {
            // Velocity BC: K' + beta*H
            sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta)
        }
        1 => {
            // Pressure BC: -(G + beta*H^T)
            -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
        }
        _ => Complex64::new(0.0, 0.0),
    };

    if self_interaction {
        match field_bc_type {
            0 => {
                coeff -= sign * (gamma * 0.5);
                if let Some(admittance) = field_elem.boundary_condition.admittance() {
                    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
                    coeff -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                }
            }
            1 => {
                coeff += sign * (beta * tau * 0.5);
            }
            _ => {}
        }
    }

    if field_bc_type == 0
        && let Some(admittance) = field_elem.boundary_condition.admittance()
    {
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
        coeff -= sign
            * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
            * admittance
            * i_omega_rho;
    }

    // Return the coordinate buffer to the thread-local pool for reuse.
    let data = element_coords.into_data();
    COORDS_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        *buf = data;
    });

    coeff
}

/// Assemble one off-diagonal H-matrix/TBEM entry derivative with respect to
/// wavenumber.
///
/// The geometry is held fixed and the regular kernel derivative is evaluated
/// analytically at the supplied quadrature points.  `beta_derivative` is
/// explicit so callers can differentiate bounded or scaled Burton--Miller
/// schedules as well as the traditional `beta(k)`.  Self interactions remain
/// on the singular-integration path and must be handled by a corresponding
/// singular derivative implementation before this helper is used for them.
#[allow(clippy::too_many_arguments)]
pub fn compute_regular_matrix_entry_wavenumber_derivative(
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    gauss_order: usize,
) -> Result<Complex64, MatrixDerivativeError> {
    if gauss_order == 0 {
        return Err(MatrixDerivativeError::InvalidQuadratureOrder);
    }
    let element_coords = get_element_coords(field_elem, nodes);
    let derivative = regular_integration_fixed_order_wavenumber_derivative(
        &source_elem.center,
        &source_elem.normal,
        &element_coords,
        field_elem.element_type,
        physics,
        gauss_order,
    );
    let value = regular_integration_fixed_order(
        &source_elem.center,
        &source_elem.normal,
        &element_coords,
        field_elem.element_type,
        physics,
        gauss_order,
    );
    let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
    let derivative_pressure_factor = physics.pressure_factor_wavenumber_derivative();
    let derivative_pressure_factor = Complex64::new(0.0, derivative_pressure_factor);
    let pressure_factor = Complex64::new(0.0, physics.pressure_factor);

    let mut coefficient = match field_bc_type {
        0 => {
            sign * (derivative.dg_dn_integral * gamma * tau
                + derivative.d2g_dnxdny_integral * beta
                + value.d2g_dnxdny_integral * beta_derivative)
        }
        1 => {
            -sign
                * (derivative.g_integral * gamma * tau
                    + derivative.dg_dnx_integral * beta
                    + value.dg_dnx_integral * beta_derivative)
        }
        _ => Complex64::new(0.0, 0.0),
    };

    if field_bc_type == 0
        && let Some(admittance) = field_elem.boundary_condition.admittance()
    {
        coefficient -= sign
            * (derivative.g_integral * gamma * tau
                + derivative.dg_dnx_integral * beta
                + value.dg_dnx_integral * beta_derivative)
            * admittance
            * pressure_factor;
        coefficient -= sign
            * (value.g_integral * gamma * tau + value.dg_dnx_integral * beta)
            * admittance
            * derivative_pressure_factor;
    }

    Ok(coefficient)
}

fn combine_adaptive_regular_matrix_entry_wavenumber_derivative(
    field_elem: &Element,
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    value: IntegrationResult,
    derivative: IntegrationResult,
) -> Complex64 {
    let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
    let pressure_factor = Complex64::new(0.0, physics.pressure_factor);
    let derivative_pressure_factor =
        Complex64::new(0.0, physics.pressure_factor_wavenumber_derivative());
    let mut coefficient = match field_bc_type {
        0 => {
            sign * (derivative.dg_dn_integral * gamma * tau
                + derivative.d2g_dnxdny_integral * beta
                + value.d2g_dnxdny_integral * beta_derivative)
        }
        1 => {
            -sign
                * (derivative.g_integral * gamma * tau
                    + derivative.dg_dnx_integral * beta
                    + value.dg_dnx_integral * beta_derivative)
        }
        _ => Complex64::new(0.0, 0.0),
    };

    if field_bc_type == 0
        && let Some(admittance) = field_elem.boundary_condition.admittance()
    {
        coefficient -= sign
            * (derivative.g_integral * gamma * tau
                + derivative.dg_dnx_integral * beta
                + value.dg_dnx_integral * beta_derivative)
            * admittance
            * pressure_factor;
        coefficient -= sign
            * (value.g_integral * gamma * tau + value.dg_dnx_integral * beta)
            * admittance
            * derivative_pressure_factor;
    }

    coefficient
}

/// Compute a fixed-geometry wavenumber derivative for a complete matrix entry.
///
/// Off-diagonal entries use the fixed-order regular path. Self-interactions
/// use the production singular/curved quadrature and include the jump terms,
/// admittance pressure factor, and Burton--Miller beta derivative. Geometry,
/// normals, boundary data, and quadrature topology are held fixed. The
/// regular off-diagonal path is the scope of the public H-matrix derivative
/// helper; this entry-point adds the singular self contract explicitly.
#[allow(clippy::too_many_arguments)]
pub fn compute_matrix_entry_wavenumber_derivative(
    source_index: usize,
    field_index: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    gauss_order: usize,
) -> Result<Complex64, MatrixDerivativeError> {
    compute_matrix_entry_wavenumber_derivative_impl(
        source_index,
        field_index,
        source_elem,
        field_elem,
        nodes,
        physics,
        gamma,
        tau,
        beta,
        beta_derivative,
        sign,
        gauss_order,
        false,
    )
}

/// Compute a derivative entry using curved/interpolated panel normals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_matrix_entry_wavenumber_derivative_curved(
    source_index: usize,
    field_index: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    gauss_order: usize,
) -> Result<Complex64, MatrixDerivativeError> {
    compute_matrix_entry_wavenumber_derivative_impl(
        source_index,
        field_index,
        source_elem,
        field_elem,
        nodes,
        physics,
        gamma,
        tau,
        beta,
        beta_derivative,
        sign,
        gauss_order,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_matrix_entry_wavenumber_derivative_impl(
    source_index: usize,
    field_index: usize,
    source_elem: &Element,
    field_elem: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    gauss_order: usize,
    curved_normals: bool,
) -> Result<Complex64, MatrixDerivativeError> {
    if gauss_order == 0 {
        return Err(MatrixDerivativeError::InvalidQuadratureOrder);
    }
    if let (Some((_, source_local)), Some((_, field_local))) = (
        decode_dof_proxy_group(source_elem.group),
        decode_dof_proxy_group(field_elem.group),
    ) {
        return Ok(compute_higher_order_matrix_entry_wavenumber_derivative(
            source_elem.dof_addresses[0],
            field_elem.dof_addresses[0],
            source_index == field_index || source_elem.connectivity == field_elem.connectivity,
            source_local,
            field_local,
            source_elem,
            field_elem,
            nodes,
            physics,
            gamma,
            tau,
            beta,
            beta_derivative,
            sign,
            gauss_order,
            curved_normals,
        ));
    }
    if source_index != field_index {
        let element_coords = get_element_coords(field_elem, nodes);
        let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
        let value = regular_integration_with_normals(
            &source_elem.center,
            &source_elem.normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            None,
            field_bc_type,
            false,
            None,
        );
        let derivative = regular_integration_wavenumber_derivative(
            &source_elem.center,
            &source_elem.normal,
            &element_coords,
            field_elem.element_type,
            field_elem.area,
            physics,
            None,
            field_bc_type,
            false,
            None,
        );
        return Ok(combine_adaptive_regular_matrix_entry_wavenumber_derivative(
            field_elem,
            physics,
            gamma,
            tau,
            beta,
            beta_derivative,
            sign,
            value,
            derivative,
        ));
    }

    let element_coords = get_element_coords(field_elem, nodes);
    let (field_bc_type, _) = get_bc_type_and_value(&field_elem.boundary_condition);
    let value = singular_integration_curved(
        &source_elem.center,
        &source_elem.normal,
        &element_coords,
        field_elem.element_type,
        physics,
        None,
        field_bc_type,
        false,
        None,
    );
    let derivative = singular_integration_wavenumber_derivative(
        &source_elem.center,
        &source_elem.normal,
        &element_coords,
        field_elem.element_type,
        physics,
        None,
        field_bc_type,
        false,
        None,
    );
    let pressure_factor = Complex64::new(0.0, physics.pressure_factor);
    let derivative_pressure_factor =
        Complex64::new(0.0, physics.pressure_factor_wavenumber_derivative());

    let mut coefficient = match field_bc_type {
        0 => {
            sign * (derivative.dg_dn_integral * gamma * tau
                + derivative.d2g_dnxdny_integral * beta
                + value.d2g_dnxdny_integral * beta_derivative)
        }
        1 => {
            -sign
                * (derivative.g_integral * gamma * tau
                    + derivative.dg_dnx_integral * beta
                    + value.dg_dnx_integral * beta_derivative)
        }
        _ => Complex64::new(0.0, 0.0),
    };

    if source_index == field_index {
        match field_bc_type {
            0 => {
                if let Some(admittance) = field_elem.boundary_condition.admittance() {
                    coefficient -= sign
                        * admittance
                        * tau
                        * 0.5
                        * (derivative_pressure_factor * beta + pressure_factor * beta_derivative);
                }
            }
            1 => {
                coefficient += sign * beta_derivative * tau * 0.5;
            }
            _ => {}
        }
    }

    if field_bc_type == 0
        && let Some(admittance) = field_elem.boundary_condition.admittance()
    {
        coefficient -= sign
            * (derivative.g_integral * gamma * tau
                + derivative.dg_dnx_integral * beta
                + value.dg_dnx_integral * beta_derivative)
            * admittance
            * pressure_factor;
        coefficient -= sign
            * (value.g_integral * gamma * tau + value.dg_dnx_integral * beta)
            * admittance
            * derivative_pressure_factor;
    }

    Ok(coefficient)
}

fn derivative_dof_owner_table(elements: &[Element]) -> Vec<usize> {
    let max_dof = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .flat_map(|element| element.dof_addresses.iter().copied())
        .max()
        .map(|dof| dof + 1)
        .unwrap_or(0);
    let mut owners = vec![usize::MAX; max_dof];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        for &dof in &element.dof_addresses {
            if dof < owners.len() {
                owners[dof] = element_index;
            }
        }
    }
    owners
}

#[allow(clippy::too_many_arguments)]
fn assemble_wavenumber_derivative_block(
    source_dofs: &[usize],
    field_dofs: &[usize],
    dof_to_element: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    beta_derivative: Complex64,
    sign: f64,
    gauss_order: usize,
) -> Result<DenseMatrix<Complex64>, MatrixDerivativeError> {
    let mut block = DenseMatrix::<Complex64>::zeros(source_dofs.len(), field_dofs.len());
    for (i_local, &source_dof) in source_dofs.iter().enumerate() {
        let source_index = dof_to_element
            .get(source_dof)
            .copied()
            .filter(|&index| index != usize::MAX)
            .ok_or(MatrixDerivativeError::MissingDofOwner { dof: source_dof })?;
        for (j_local, &field_dof) in field_dofs.iter().enumerate() {
            let field_index = dof_to_element
                .get(field_dof)
                .copied()
                .filter(|&index| index != usize::MAX)
                .ok_or(MatrixDerivativeError::MissingDofOwner { dof: field_dof })?;
            block[[i_local, j_local]] = compute_matrix_entry_wavenumber_derivative(
                source_index,
                field_index,
                &elements[source_index],
                &elements[field_index],
                nodes,
                physics,
                gamma,
                tau,
                beta,
                beta_derivative,
                sign,
                gauss_order,
            )?;
        }
    }
    Ok(block)
}

fn exact_constant_panel_dof_count(elements: &[Element]) -> Result<usize, String> {
    let mut num_dofs = 0usize;
    for (index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        if element.dof_addresses.len() != 1 {
            return Err(format!(
                "exact TBEM reference supports constant-panel layouts only; element {index} has {} DOFs",
                element.dof_addresses.len()
            ));
        }
        num_dofs = num_dofs
            .checked_add(1)
            .ok_or_else(|| "exact TBEM reference DOF count overflow".to_string())?;
    }
    Ok(num_dofs)
}

/// Apply the exact, uncompressed stabilized constant-panel TBEM operator.
///
/// This performs an O(N²) kernel application but stores only O(N) values. It
/// is primarily intended for true-residual verification of compressed or FMM
/// solves, where assembling a dense reference matrix would defeat the memory
/// benefit of the approximate operator.
pub fn apply_tbem_stabilized_exact(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let num_dofs = exact_constant_panel_dof_count(elements)?;
    if x.len() != num_dofs {
        return Err(format!(
            "exact TBEM matvec expected {num_dofs} values, got {}",
            x.len()
        ));
    }
    if !sign.is_finite() || !beta_scale.is_finite() {
        return Err("exact TBEM matvec requires finite sign and beta scale".into());
    }

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    // Exact matvec rows (rayon above the machine-adaptive gate, serial for
    // small systems). Work is source × field kernel evaluations.
    let rows: Vec<(usize, Complex64)> = gated_enumerate_filter_map(
        elements,
        num_dofs.saturating_mul(elements.len()),
        BEM_ENTRIES_PER_THREAD,
        |source_index, source_element| {
            if source_element.property.is_evaluation() {
                return None;
            }
            let source_dof = source_element.dof_addresses[0];
            let value = elements
                .iter()
                .enumerate()
                .filter(|(_, field_element)| !field_element.property.is_evaluation())
                .map(|(field_index, field_element)| {
                    let field_dof = field_element.dof_addresses[0];
                    compute_matrix_entry(
                        source_index,
                        field_index,
                        source_element,
                        field_element,
                        nodes,
                        physics,
                        gamma,
                        tau,
                        beta,
                        sign,
                    ) * x[field_dof]
                })
                .sum();
            Some((source_dof, value))
        },
    );

    let mut output = vec![Complex64::new(0.0, 0.0); num_dofs];
    for (dof, value) in rows {
        output[dof] += value;
    }
    Ok(output)
}

/// Apply the exact fixed-geometry wavenumber derivative of the stabilized
/// constant-panel TBEM operator.
///
/// This is intentionally an O(N²), O(N) reference operator for validating
/// compressed and FMM derivative paths. It uses the complete matrix-entry
/// derivative, including singular self interactions; `beta_scale` is held
/// fixed and its derivative is propagated through beta.
pub fn apply_tbem_stabilized_exact_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let num_dofs = exact_constant_panel_dof_count(elements)?;
    if x.len() != num_dofs {
        return Err(format!(
            "exact TBEM derivative matvec expected {num_dofs} values, got {}",
            x.len()
        ));
    }
    if !sign.is_finite() || !beta_scale.is_finite() {
        return Err("exact TBEM derivative matvec requires finite sign and beta scale".into());
    }

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let beta_derivative = physics.burton_miller_beta_wavenumber_derivative() * beta_scale;
    // Exact derivative matvec rows (rayon above the machine-adaptive gate,
    // serial for small systems).
    let rows: Vec<(usize, Complex64)> = gated_enumerate_filter_map(
        elements,
        num_dofs.saturating_mul(elements.len()),
        BEM_ENTRIES_PER_THREAD,
        |source_index, source_element| {
            if source_element.property.is_evaluation() {
                return None;
            }
            let source_dof = source_element.dof_addresses[0];
            let value = elements
                .iter()
                .enumerate()
                .filter(|(_, field_element)| !field_element.property.is_evaluation())
                .map(|(field_index, field_element)| {
                    let field_dof = field_element.dof_addresses[0];
                    let derivative = compute_matrix_entry_wavenumber_derivative(
                        source_index,
                        field_index,
                        source_element,
                        field_element,
                        nodes,
                        physics,
                        gamma,
                        tau,
                        beta,
                        beta_derivative,
                        sign,
                        6,
                    )
                    .expect("positive internal derivative quadrature order");
                    derivative * x[field_dof]
                })
                .sum();
            Some((source_dof, value))
        },
    );

    let mut output = vec![Complex64::new(0.0, 0.0); num_dofs];
    for (dof, value) in rows {
        output[dof] += value;
    }
    Ok(output)
}

/// Apply the exact, uncompressed Hermitian (conjugate transpose) of the
/// stabilized constant-panel TBEM operator: `y = Aᴴ x`.
///
/// Same O(N²) entry evaluations as [`apply_tbem_stabilized_exact`] with
/// conjugated coefficients accumulated by column instead of by row; stores
/// only O(N) values. Intended for true-residual verification of adjoint
/// solves against compressed operators.
pub fn apply_tbem_stabilized_exact_hermitian(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let num_dofs = exact_constant_panel_dof_count(elements)?;
    if x.len() != num_dofs {
        return Err(format!(
            "exact TBEM hermitian matvec expected {num_dofs} values, got {}",
            x.len()
        ));
    }
    if !sign.is_finite() || !beta_scale.is_finite() {
        return Err("exact TBEM hermitian matvec requires finite sign and beta scale".into());
    }

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    // (Aᴴ x)_j = Σ_i conj(A[i, j]) · x_i: parallel over the output (field)
    // index, inner loop over sources.
    // Exact adjoint matvec columns (rayon above the machine-adaptive gate,
    // serial for small systems).
    let columns: Vec<(usize, Complex64)> = gated_enumerate_filter_map(
        elements,
        num_dofs.saturating_mul(elements.len()),
        BEM_ENTRIES_PER_THREAD,
        |field_index, field_element| {
            if field_element.property.is_evaluation() {
                return None;
            }
            let field_dof = field_element.dof_addresses[0];
            let value = elements
                .iter()
                .enumerate()
                .filter(|(_, source_element)| !source_element.property.is_evaluation())
                .map(|(source_index, source_element)| {
                    let source_dof = source_element.dof_addresses[0];
                    compute_matrix_entry(
                        source_index,
                        field_index,
                        source_element,
                        field_element,
                        nodes,
                        physics,
                        gamma,
                        tau,
                        beta,
                        sign,
                    )
                    .conj()
                        * x[source_dof]
                })
                .sum();
            Some((field_dof, value))
        },
    );

    let mut output = vec![Complex64::new(0.0, 0.0); num_dofs];
    for (dof, value) in columns {
        output[dof] += value;
    }
    Ok(output)
}

/// Apply an exact, uncompressed stabilized TBEM operator for panel-local
/// linear or quadratic basis functions.
///
/// This is the higher-order counterpart of [`apply_tbem_stabilized_exact`].
/// It evaluates the same basis-weighted entry kernel used by the production
/// H-matrix builder, but retains only the output vector. The routine supports
/// panel-local and shared global DOF layouts. Shared DOFs are
/// accumulated once for every field/source incidence, matching the global
/// continuous assembly contract; all referenced global DOFs must still form
/// one contiguous range.
pub fn apply_tbem_stabilized_exact_higher_order(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, false, false, false,
    )
}

/// Apply the exact curved-normal higher-order TBEM action using one
/// collocation row per shared global DOF. This matches the owner-row
/// convention used by the basis-aware MLFMM backend for continuous layouts.
pub fn apply_tbem_stabilized_exact_higher_order_owner(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, false, true, true,
    )
}

/// Apply exact flat-panel higher-order TBEM action using one collocation row
/// per shared global DOF. This is the continuous MLFMM reference matching the
/// flat-panel continuous TBEM convention.
pub fn apply_tbem_stabilized_exact_higher_order_owner_flat(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, false, false, true,
    )
}

/// Apply the exact Hermitian-transposed higher-order stabilized TBEM action.
pub fn apply_tbem_stabilized_exact_higher_order_hermitian(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, true, false, false,
    )
}

/// Apply the exact curved-normal Hermitian-transposed higher-order TBEM action
/// using one collocation row per shared global DOF (the continuous MLFMM
/// convention).
pub fn apply_tbem_stabilized_exact_higher_order_owner_hermitian(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, true, true, true,
    )
}

/// Apply the Hermitian-transposed flat-panel higher-order TBEM action using
/// one collocation row per shared global DOF.
pub fn apply_tbem_stabilized_exact_higher_order_owner_flat_hermitian(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, true, false, true,
    )
}

/// Apply an exact higher-order TBEM action using curved nodal normals.
///
/// This variant mirrors the basis-weighted SLFMM/MLFMM near-field kernels,
/// which use each collocation normal and the field element's interpolated
/// nodal normals. The default exact reference remains affine so it preserves
/// parity with the dense/TBEM and H-matrix higher-order paths.
pub fn apply_tbem_stabilized_exact_higher_order_curved(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, false, true, false,
    )
}

/// Apply the exact Hermitian-transposed higher-order curved-normal TBEM action.
pub fn apply_tbem_stabilized_exact_higher_order_curved_hermitian(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    apply_tbem_stabilized_exact_higher_order_impl(
        elements, nodes, physics, sign, beta_scale, x, true, true, false,
    )
}

fn exact_higher_order_dof_count(elements: &[Element]) -> Result<usize, String> {
    let mut addresses = std::collections::HashSet::new();
    let mut max_dof = None;
    for (index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        if element.dof_addresses.len() <= 1 {
            return Err(format!(
                "exact higher-order TBEM reference requires linear or quadratic panels; element {index} has {} DOFs",
                element.dof_addresses.len()
            ));
        }
        if basis_order_for_dof_count(element, element.dof_addresses.len()).is_none() {
            return Err(format!(
                "exact higher-order TBEM reference does not support element {index} with {:?} and {} DOFs",
                element.element_type,
                element.dof_addresses.len()
            ));
        }
        for &dof in &element.dof_addresses {
            addresses.insert(dof);
            max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
        }
    }
    let count = match max_dof {
        Some(dof) => dof
            .checked_add(1)
            .ok_or_else(|| "exact higher-order TBEM reference DOF count overflow".to_string())?,
        None => 0,
    };
    if addresses.len() != count {
        return Err(
            "exact higher-order TBEM reference requires contiguous global DOFs".to_string(),
        );
    }
    Ok(count)
}

fn apply_tbem_stabilized_exact_higher_order_impl(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    x: &[Complex64],
    hermitian: bool,
    curved_normals: bool,
    owner_rows: bool,
) -> Result<Vec<Complex64>, String> {
    let num_dofs = exact_higher_order_dof_count(elements)?;
    if x.len() != num_dofs {
        return Err(format!(
            "exact higher-order TBEM matvec expected {num_dofs} values, got {}",
            x.len()
        ));
    }
    if !sign.is_finite() || !beta_scale.is_finite() {
        return Err("exact higher-order TBEM matvec requires finite sign and beta scale".into());
    }

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let active: Vec<(usize, &Element)> = elements
        .iter()
        .enumerate()
        .filter(|(_, element)| !element.property.is_evaluation())
        .collect();
    let active_ref = &active;
    let mut row_owners = vec![None; num_dofs];
    for &(element_index, element) in active_ref {
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            if row_owners[dof].is_none() {
                row_owners[dof] = Some((element_index, local));
            }
        }
    }
    let row_owners_ref = &row_owners;

    // Higher-order exact matvec rows (rayon above the machine-adaptive gate,
    // serial for small systems). Work is active elements × DOFs kernel
    // evaluations.
    let active_work = active_ref.len().saturating_mul(num_dofs);
    let rows: Vec<(usize, Complex64)> = if hermitian {
        gated_flat_map(
            active_ref,
            active_work,
            BEM_ENTRIES_PER_THREAD,
            |&(field_index, field_element)| {
                field_element
                    .dof_addresses
                    .iter()
                    .enumerate()
                    .map(move |(field_local, &field_dof)| {
                        let value = active_ref
                            .iter()
                            .map(|&(source_index, source_element)| {
                                source_element
                                    .dof_addresses
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(source_local, &source_dof)| {
                                        if owner_rows
                                            && row_owners_ref[source_dof]
                                                != Some((source_index, source_local))
                                        {
                                            return None;
                                        }
                                        Some(
                                            compute_higher_order_matrix_entry_with_normals(
                                                source_dof,
                                                field_dof,
                                                source_index == field_index
                                                    || source_element.connectivity
                                                        == field_element.connectivity,
                                                source_local,
                                                field_local,
                                                source_element,
                                                field_element,
                                                nodes,
                                                physics,
                                                gamma,
                                                tau,
                                                beta,
                                                sign,
                                                curved_normals,
                                            )
                                            .conj()
                                                * x[source_dof],
                                        )
                                    })
                                    .sum::<Complex64>()
                            })
                            .sum::<Complex64>();
                        (field_dof, value)
                    })
                    .collect::<Vec<_>>()
            },
        )
    } else {
        gated_flat_map(
            active_ref,
            active_work,
            BEM_ENTRIES_PER_THREAD,
            |&(source_index, source_element)| {
                source_element
                    .dof_addresses
                    .iter()
                    .enumerate()
                    .filter_map(move |(source_local, &source_dof)| {
                        if owner_rows
                            && row_owners_ref[source_dof] != Some((source_index, source_local))
                        {
                            return None;
                        }
                        let value = active_ref
                            .iter()
                            .map(|&(field_index, field_element)| {
                                field_element
                                    .dof_addresses
                                    .iter()
                                    .enumerate()
                                    .map(|(field_local, &field_dof)| {
                                        compute_higher_order_matrix_entry_with_normals(
                                            source_dof,
                                            field_dof,
                                            source_index == field_index
                                                || source_element.connectivity
                                                    == field_element.connectivity,
                                            source_local,
                                            field_local,
                                            source_element,
                                            field_element,
                                            nodes,
                                            physics,
                                            gamma,
                                            tau,
                                            beta,
                                            sign,
                                            curved_normals,
                                        ) * x[field_dof]
                                    })
                                    .sum::<Complex64>()
                            })
                            .sum::<Complex64>();
                        Some((source_dof, value))
                    })
                    .collect::<Vec<_>>()
            },
        )
    };

    let mut output = vec![Complex64::new(0.0, 0.0); num_dofs];
    for (dof, value) in rows {
        output[dof] += value;
    }
    Ok(output)
}

// ============================================================================
// Partially-pivoted ACA algorithm
// ============================================================================

/// Perform partially-pivoted Adaptive Cross Approximation on a single far-field block.
///
/// The ACA algorithm generates a low-rank approximation `A ≈ U V^T` without ever
/// forming the full block matrix. It only evaluates individual matrix entries
/// on demand, making it kernel-independent and efficient.
///
/// # Arguments
/// * `source_indices` - Global element indices for the source cluster
/// * `field_indices` - Global element indices for the field cluster
/// * `elements` - All mesh elements
/// * `nodes` - Node coordinates
/// * `physics` - Physics parameters
/// * `gamma`, `tau`, `beta` - Burton-Miller formulation parameters
/// * `dg_dn_sign` - Sign factor for dG/dn based on formulation switching
/// * `aca_tolerance` - Convergence tolerance (typically 1e-4)
/// * `max_rank` - Maximum allowed rank
///
/// # Returns
/// Tuple of `(U, V, rank, converged)` where `A ≈ U V^T` (no conjugation).
/// `converged` is
/// `true` when ACA stopped because the Frobenius criterion was met (or the
/// block was exhausted exactly), and `false` when the loop terminated solely
/// because the user-supplied `max_rank` cap was binding — in that case the
/// approximation may be under-resolved.
#[cfg(test)]
fn aca_partial_pivot(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    dg_dn_sign: f64,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    aca_partial_pivot_with_mode(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        MatrixEntryMode::Value {
            gamma,
            tau,
            beta,
            sign: dg_dn_sign,
        },
        aca_tolerance,
        max_rank,
    )
}

fn aca_partial_pivot_with_mode(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    entry_mode: MatrixEntryMode,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    let m = source_indices.len(); // number of source DOFs (rows)
    let n = field_indices.len(); // number of field DOFs (columns)

    if m == 0 || n == 0 {
        return (
            DenseMatrix::<Complex64>::zeros(m, 0),
            DenseMatrix::<Complex64>::zeros(n, 0),
            0,
            true,
        );
    }

    let effective_max_rank = max_rank.min(m).min(n);
    // The user-supplied cap is the binding constraint iff it is strictly
    // tighter than what the geometry permits. When `max_rank >= min(m, n)`
    // the loop is effectively unbounded by the user cap; full-rank exhaustion
    // in that regime is exact and counts as converged.
    let cap_is_user_supplied = max_rank < m.min(n);
    let mut converged = false;

    // Columns of U and V accumulated during ACA iterations
    let mut u_cols: Vec<Vec<Complex64>> = Vec::with_capacity(effective_max_rank);
    let mut v_cols: Vec<Vec<Complex64>> = Vec::with_capacity(effective_max_rank);

    // Track which rows have been used as pivots
    let mut used_rows = vec![false; m];

    // Accumulate Frobenius norm squared of the approximation
    let mut frobenius_sq = 0.0_f64;

    // Pick initial pivot row using a deterministic hash of block indices
    // to avoid sensitivity to row 0 being in a shadowed/degenerate region.
    let mut pivot_row = if m > 1 {
        let seed = source_indices[0]
            .wrapping_mul(2654435761)
            .wrapping_add(field_indices[0]);
        seed % m
    } else {
        0
    };

    for k in 0..effective_max_rank {
        // Step 1: Compute row `pivot_row` of the residual block
        // R[pivot_row, :] = A[pivot_row, :] - sum_{l<k} u_l[pivot_row] * v_l^H
        let mut row_residual = vec![Complex64::new(0.0, 0.0); n];
        for (j_local, &j_global) in field_indices.iter().enumerate() {
            row_residual[j_local] = entry_mode.evaluate(
                source_indices[pivot_row],
                j_global,
                &elements[source_indices[pivot_row]],
                &elements[j_global],
                nodes,
                physics,
            );
        }
        // Subtract contributions from previous low-rank terms
        for l in 0..k {
            let u_val = u_cols[l][pivot_row];
            for j_local in 0..n {
                row_residual[j_local] -= u_val * v_cols[l][j_local];
            }
        }

        // Step 2: Find the column pivot (max absolute value)
        let mut max_val = 0.0_f64;
        let mut pivot_col = 0_usize;
        for (j, residual) in row_residual.iter().enumerate().take(n) {
            let val = residual.norm();
            if val > max_val {
                max_val = val;
                pivot_col = j;
            }
        }

        // If the row residual is essentially zero, this block is already well-approximated
        if max_val < 1e-15 {
            converged = true;
            break;
        }

        // Step 3: Set v_new = row_residual / row_residual[pivot_col]
        let pivot_val = row_residual[pivot_col];
        let v_new: Vec<Complex64> = row_residual.iter().map(|&x| x / pivot_val).collect();

        // Step 4: Compute column `pivot_col` of the residual block
        // R[:, pivot_col] = A[:, pivot_col] - sum_{l<k} v_l[pivot_col] * u_l
        let mut col_residual = vec![Complex64::new(0.0, 0.0); m];
        for (i_local, &i_global) in source_indices.iter().enumerate() {
            col_residual[i_local] = entry_mode.evaluate(
                i_global,
                field_indices[pivot_col],
                &elements[i_global],
                &elements[field_indices[pivot_col]],
                nodes,
                physics,
            );
        }
        // Subtract contributions from previous low-rank terms
        for l in 0..k {
            let v_val = v_cols[l][pivot_col];
            for i_local in 0..m {
                col_residual[i_local] -= v_val * u_cols[l][i_local];
            }
        }

        let u_new = col_residual;

        // Step 5: Check convergence using Frobenius norm criterion
        let u_norm_sq: f64 = u_new.iter().map(|x| x.norm_sqr()).sum();
        let v_norm_sq: f64 = v_new.iter().map(|x| x.norm_sqr()).sum();
        let delta = u_norm_sq.sqrt() * v_norm_sq.sqrt();

        // Update ‖U V^T‖_F² with the full formula including cross-terms.
        // For A = U V^T (no conjugation in the outer-product sum),
        //   ‖A‖_F² = Σ_{l,m} <u_l, u_m> · <v_l, v_m>
        // where <a, b> = Σ_i a_i · conj(b_i). The l=m terms give
        // ‖u_l‖² · ‖v_l‖² and the cross terms l≠m collapse to
        // 2 · Re(<u_k, u_l> · <v_k, v_l>) by symmetry.
        frobenius_sq += u_norm_sq * v_norm_sq;
        for l in 0..k {
            let u_dot: Complex64 = u_new
                .iter()
                .zip(u_cols[l].iter())
                .map(|(a, b)| a * b.conj())
                .sum();
            let v_dot: Complex64 = v_new
                .iter()
                .zip(v_cols[l].iter())
                .map(|(a, b)| a * b.conj())
                .sum();
            frobenius_sq += 2.0 * (u_dot * v_dot).re;
        }

        u_cols.push(u_new);
        v_cols.push(v_new);
        used_rows[pivot_row] = true;

        // Check convergence: ‖u_k‖ · ‖v_k‖ < tol · ‖U V^T‖_F
        let frobenius_norm = frobenius_sq.sqrt();
        if delta < aca_tolerance * frobenius_norm && k > 0 {
            converged = true;
            break;
        }

        // Step 6: Pick next pivot row = row with max |u_new[i]| among unused rows.
        // If all rows have been used, we've reached full rank — break early.
        let last_u = &u_cols[u_cols.len() - 1];
        let mut max_u_val = 0.0_f64;
        let mut found_unused = false;
        let mut next_pivot = 0_usize;
        for i in 0..m {
            if !used_rows[i] {
                let val = last_u[i].norm();
                if !found_unused || val > max_u_val {
                    max_u_val = val;
                    next_pivot = i;
                    found_unused = true;
                }
            }
        }
        if !found_unused {
            // All m rows have been used as pivots — the partial-pivot ACA has
            // reproduced the block exactly within its row span.
            converged = true;
            break;
        }
        pivot_row = next_pivot;
    }

    let rank = u_cols.len();
    // If the loop never set `converged` (i.e. it exited because k reached
    // `effective_max_rank`), the block is "saturated" only when the user
    // cap was the binding constraint. Geometric saturation already implies
    // the approximation is exact, so it counts as converged.
    if !converged && !cap_is_user_supplied {
        converged = true;
    }

    // Assemble U and V matrices from the column vectors
    let mut u_matrix = DenseMatrix::<Complex64>::zeros(m, rank);
    let mut v_matrix = DenseMatrix::<Complex64>::zeros(n, rank);

    for (k, u_col) in u_cols.iter().enumerate() {
        for i in 0..m {
            u_matrix[[i, k]] = u_col[i];
        }
    }
    for (k, v_col) in v_cols.iter().enumerate() {
        for j in 0..n {
            v_matrix[[j, k]] = v_col[j];
        }
    }

    (u_matrix, v_matrix, rank, converged)
}

// ============================================================================
// ACA+ (Bebendorf–Rjasanow reference-column pivot rule)
// ============================================================================

/// Low-rank compression variant selector for `build_hmatrix_with_variant`.
///
/// * `PartialPivot` — textbook partially-pivoted ACA. Cheapest per iteration;
///   prone to silent under-resolution on sign-changing kernels.
/// * `Plus` — Bebendorf–Rjasanow ACA+ with a tracked reference column. The
///   row pivot is anchored against a column known to carry residual energy.
/// * `Gp` — ACA-GP: ACA+ with a geometric (rather than hash-based)
///   initialisation of the reference column and the first pivot row, anchored
///   to the cluster centroids. The doc's "cheap upgrade" toward HCA
///   (`bem-step1.md` §1.1, Yastrebov 2025).
/// * `Hca` — interpolation-seeded HCA. A small Chebyshev-like geometric sample
///   from both clusters selects a robust initial cross before ACA+ refinement.
///   This is the production default for Stage 1: if a block still hits the rank
///   cap, the builder stores it dense rather than accepting under-resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcaVariant {
    /// Original partially-pivoted ACA (single residual stream).
    PartialPivot,
    /// ACA+ with a tracked reference column.
    Plus,
    /// ACA-GP — ACA+ with geometric pivot/reference initialisation.
    Gp,
    /// HCA — interpolation-seeded ACA+ refinement.
    #[default]
    Hca,
}

/// Geometric "anchor" — pick the local index whose element centroid is
/// closest to the centroid of `indices`. Used by ACA-GP to seed the
/// reference column and initial pivot row from cluster geometry instead of
/// hashing block identifiers.
fn centroid_anchor(indices: &[usize], elements: &[Element]) -> usize {
    if indices.len() <= 1 {
        return 0;
    }
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for &gi in indices {
        cx += elements[gi].center[0];
        cy += elements[gi].center[1];
        cz += elements[gi].center[2];
    }
    let n = indices.len() as f64;
    cx /= n;
    cy /= n;
    cz /= n;

    let mut best_local = 0_usize;
    let mut best_d2 = f64::INFINITY;
    for (local, &gi) in indices.iter().enumerate() {
        let dx = elements[gi].center[0] - cx;
        let dy = elements[gi].center[1] - cy;
        let dz = elements[gi].center[2] - cz;
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best_local = local;
        }
    }
    best_local
}

/// ACA+ — partially-pivoted ACA augmented with a reference column.
///
/// At every iteration the row pivot is selected from the residual evaluated
/// at a fixed reference column `j_ref`, which is updated whenever it would
/// collide with the chosen column pivot. This guarantees the row pivot is
/// chosen against a column known to carry residual energy, which prevents the
/// "silent zero" failure mode that partial-pivot ACA exhibits on
/// sign-changing kernels.
///
/// Per-iteration kernel-eval cost is approximately `n + 2 m` (one extra
/// column residual versus partial-pivot ACA), with much improved robustness.
///
/// Conjugation convention follows the rest of this file: `A ≈ U V^T` (no
/// Hermitian conjugation in the outer-product sum). `HMatrix::matvec` and
/// `apply_transpose` consume `(U, V)` under the same convention.
#[cfg(test)]
fn aca_plus(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    dg_dn_sign: f64,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    let n = field_indices.len();
    let initial_j_ref = if n > 1 {
        // Deterministic hash of block identifiers — preserves the original
        // seed behaviour for the ACA+ variant.
        let seed = source_indices[0]
            .wrapping_mul(2246822519)
            .wrapping_add(field_indices[0].wrapping_mul(3266489917));
        seed % n
    } else {
        0
    };
    aca_plus_with_seed(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        MatrixEntryMode::Value {
            gamma,
            tau,
            beta,
            sign: dg_dn_sign,
        },
        aca_tolerance,
        max_rank,
        initial_j_ref,
        None,
    )
}

fn aca_plus_with_mode(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    entry_mode: MatrixEntryMode,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    let n = field_indices.len();
    let initial_j_ref = if n > 1 {
        let seed = source_indices[0]
            .wrapping_mul(2246822519)
            .wrapping_add(field_indices[0].wrapping_mul(3266489917));
        seed % n
    } else {
        0
    };
    aca_plus_with_seed(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        entry_mode,
        aca_tolerance,
        max_rank,
        initial_j_ref,
        None,
    )
}

/// ACA-GP — ACA+ with the reference column and the first pivot row chosen
/// from cluster geometry instead of an identifier hash. Closer to HCA in
/// spirit (interpolation-anchored seeding) without requiring kernel-
/// interpolation infrastructure. See `bem-step1.md` §1.1.
#[cfg(test)]
fn aca_gp(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    dg_dn_sign: f64,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    let initial_j_ref = if field_indices.len() > 1 {
        centroid_anchor(field_indices, elements)
    } else {
        0
    };
    let initial_pivot_row = if source_indices.len() > 1 {
        Some(centroid_anchor(source_indices, elements))
    } else {
        None
    };
    aca_plus_with_seed(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        MatrixEntryMode::Value {
            gamma,
            tau,
            beta,
            sign: dg_dn_sign,
        },
        aca_tolerance,
        max_rank,
        initial_j_ref,
        initial_pivot_row,
    )
}

fn aca_gp_with_mode(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    entry_mode: MatrixEntryMode,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    let initial_j_ref = if field_indices.len() > 1 {
        centroid_anchor(field_indices, elements)
    } else {
        0
    };
    let initial_pivot_row = if source_indices.len() > 1 {
        Some(centroid_anchor(source_indices, elements))
    } else {
        None
    };
    aca_plus_with_seed(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        entry_mode,
        aca_tolerance,
        max_rank,
        initial_j_ref,
        initial_pivot_row,
    )
}

/// Local interpolation stencil for HCA seeding.
///
/// The continuous HCA literature seeds ACA from interpolation points. On a
/// surface mesh block the available interpolation nodes are element
/// collocation centers, so we pick a deterministic Chebyshev-like stencil:
/// centroid anchor plus extrema along each coordinate axis. This avoids the
/// hash-sensitive start of plain ACA+ and samples the block geometry before
/// refinement.
fn interpolation_seed_indices(indices: &[usize], elements: &[Element]) -> Vec<usize> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut seeds = vec![centroid_anchor(indices, elements)];
    for axis in 0..3 {
        let mut min_local = 0_usize;
        let mut max_local = 0_usize;
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        for (local, &global) in indices.iter().enumerate() {
            let value = elements[global].center[axis];
            if value < min_value {
                min_value = value;
                min_local = local;
            }
            if value > max_value {
                max_value = value;
                max_local = local;
            }
        }
        seeds.push(min_local);
        seeds.push(max_local);
    }

    seeds.sort_unstable();
    seeds.dedup();
    seeds
}

/// Interpolation-seeded Hybrid Cross Approximation.
///
/// HCA starts from an interpolation stencil instead of a single arbitrary row.
/// We evaluate the small row/column seed tensor, choose the largest-magnitude
/// cross as the first pivot/reference pair, then refine with ACA+. The builder
/// treats non-convergence as a request for dense storage, so HCA never silently
/// accepts a rank-capped far-field block.
#[cfg(test)]
fn hca(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    dg_dn_sign: f64,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    hca_with_mode(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        MatrixEntryMode::Value {
            gamma,
            tau,
            beta,
            sign: dg_dn_sign,
        },
        aca_tolerance,
        max_rank,
    )
}

fn hca_with_mode(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    entry_mode: MatrixEntryMode,
    aca_tolerance: f64,
    max_rank: usize,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    if source_indices.is_empty() || field_indices.is_empty() {
        return (
            DenseMatrix::<Complex64>::zeros(source_indices.len(), 0),
            DenseMatrix::<Complex64>::zeros(field_indices.len(), 0),
            0,
            true,
        );
    }

    let source_seeds = interpolation_seed_indices(source_indices, elements);
    let field_seeds = interpolation_seed_indices(field_indices, elements);

    let mut best_row = source_seeds[0];
    let mut best_col = field_seeds[0];
    let mut best_norm = 0.0_f64;

    for &i_local in &source_seeds {
        for &j_local in &field_seeds {
            let value = entry_mode.evaluate(
                source_indices[i_local],
                field_indices[j_local],
                &elements[source_indices[i_local]],
                &elements[field_indices[j_local]],
                nodes,
                physics,
            );
            let norm = value.norm();
            if norm > best_norm {
                best_norm = norm;
                best_row = i_local;
                best_col = j_local;
            }
        }
    }

    aca_plus_with_seed(
        source_indices,
        field_indices,
        elements,
        nodes,
        physics,
        entry_mode,
        aca_tolerance,
        max_rank,
        best_col,
        Some(best_row),
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_dense_block_matrix_with_mode(
    src_dofs: &[usize],
    fld_dofs: &[usize],
    src_element_indices: &[usize],
    fld_element_indices: &[usize],
    dof_to_element: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    entry_mode: MatrixEntryMode,
) -> DenseMatrix<Complex64> {
    let mut block_matrix = DenseMatrix::<Complex64>::zeros(src_dofs.len(), fld_dofs.len());
    let use_incidence_elements = src_element_indices.len() == src_dofs.len()
        && fld_element_indices.len() == fld_dofs.len()
        && src_element_indices.iter().all(|&index| {
            elements
                .get(index)
                .is_some_and(|element| has_shared_dof_proxy_group(element.group))
        })
        && fld_element_indices.iter().all(|&index| {
            elements
                .get(index)
                .is_some_and(|element| has_shared_dof_proxy_group(element.group))
        });
    for (i_local, &i_global_dof) in src_dofs.iter().enumerate() {
        let src_elem_idx = if use_incidence_elements {
            src_element_indices[i_local]
        } else {
            dof_to_element[i_global_dof]
        };
        assert_ne!(
            src_elem_idx,
            usize::MAX,
            "build_hmatrix: source DOF {i_global_dof} has no owning element \
            (mesh/DOF numbering inconsistency)"
        );
        let src_local = if use_incidence_elements {
            0
        } else {
            elements[src_elem_idx]
                .dof_addresses
                .iter()
                .position(|&dof| dof == i_global_dof)
                .expect("build_hmatrix: source DOF owner does not contain DOF")
        };
        for (j_local, &j_global_dof) in fld_dofs.iter().enumerate() {
            let fld_elem_idx = if use_incidence_elements {
                fld_element_indices[j_local]
            } else {
                dof_to_element[j_global_dof]
            };
            assert_ne!(
                fld_elem_idx,
                usize::MAX,
                "build_hmatrix: field DOF {j_global_dof} has no owning element"
            );
            let fld_local = if use_incidence_elements {
                0
            } else {
                elements[fld_elem_idx]
                    .dof_addresses
                    .iter()
                    .position(|&dof| dof == j_global_dof)
                    .expect("build_hmatrix: field DOF owner does not contain DOF")
            };
            block_matrix[[i_local, j_local]] = entry_mode.evaluate_dof(
                i_global_dof,
                j_global_dof,
                src_elem_idx,
                fld_elem_idx,
                src_local,
                fld_local,
                &elements[src_elem_idx],
                &elements[fld_elem_idx],
                nodes,
                physics,
            );
        }
    }
    block_matrix
}

/// Common ACA+ kernel parameterised by initial reference column and (for
/// ACA-GP) an initial pivot row hint. The pivot-row hint is consulted on
/// the very first iteration only when the residual at the reference column
/// is essentially uniform — otherwise the residual-norm-based pivot wins.
fn aca_plus_with_seed(
    source_indices: &[usize],
    field_indices: &[usize],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    entry_mode: MatrixEntryMode,
    aca_tolerance: f64,
    max_rank: usize,
    initial_j_ref: usize,
    initial_pivot_row_hint: Option<usize>,
) -> (DenseMatrix<Complex64>, DenseMatrix<Complex64>, usize, bool) {
    let m = source_indices.len();
    let n = field_indices.len();

    if m == 0 || n == 0 {
        return (
            DenseMatrix::<Complex64>::zeros(m, 0),
            DenseMatrix::<Complex64>::zeros(n, 0),
            0,
            true,
        );
    }

    let effective_max_rank = max_rank.min(m).min(n);
    let cap_is_user_supplied = max_rank < m.min(n);
    let mut converged = false;

    let mut u_cols: Vec<Vec<Complex64>> = Vec::with_capacity(effective_max_rank);
    let mut v_cols: Vec<Vec<Complex64>> = Vec::with_capacity(effective_max_rank);

    let mut used_rows = vec![false; m];
    let mut used_cols = vec![false; n];

    let mut j_ref = initial_j_ref.min(n.saturating_sub(1));
    let pivot_row_hint = initial_pivot_row_hint.map(|h| h.min(m.saturating_sub(1)));
    let mut first_iteration = true;

    let mut frobenius_sq = 0.0_f64;

    let column_residual =
        |j_local: usize, u_cols: &[Vec<Complex64>], v_cols: &[Vec<Complex64>]| -> Vec<Complex64> {
            let j_global = field_indices[j_local];
            let mut col = vec![Complex64::new(0.0, 0.0); m];
            for (i_local, &i_global) in source_indices.iter().enumerate() {
                col[i_local] = entry_mode.evaluate(
                    i_global,
                    j_global,
                    &elements[i_global],
                    &elements[j_global],
                    nodes,
                    physics,
                );
            }
            for (u_l, v_l) in u_cols.iter().zip(v_cols.iter()) {
                let v_val = v_l[j_local];
                for i in 0..m {
                    col[i] -= v_val * u_l[i];
                }
            }
            col
        };

    let row_residual =
        |i_local: usize, u_cols: &[Vec<Complex64>], v_cols: &[Vec<Complex64>]| -> Vec<Complex64> {
            let i_global = source_indices[i_local];
            let mut row = vec![Complex64::new(0.0, 0.0); n];
            for (j_local, &j_global) in field_indices.iter().enumerate() {
                row[j_local] = entry_mode.evaluate(
                    i_global,
                    j_global,
                    &elements[i_global],
                    &elements[j_global],
                    nodes,
                    physics,
                );
            }
            for (u_l, v_l) in u_cols.iter().zip(v_cols.iter()) {
                let u_val = u_l[i_local];
                for j in 0..n {
                    row[j] -= u_val * v_l[j];
                }
            }
            row
        };

    for _k in 0..effective_max_rank {
        // Step 1: residual at the reference column. This guides the choice of
        // row pivot — the key ACA+ idea.
        let mut r_col = column_residual(j_ref, &u_cols, &v_cols);

        // Pick row pivot among unused rows; if the reference column itself
        // is exhausted, hop to a different unused column and try again.
        let mut pivot_row = 0_usize;
        let mut max_val = 0.0_f64;
        let mut found = false;

        // ACA-GP geometric seed: on the very first iteration, if the
        // centroid-anchored hint row carries non-trivial residual mass, prefer
        // it over argmax. This anchors the initial outer product to the
        // geometrically-central element of the source cluster, which spreads
        // subsequent pivots better than a hash-seeded start row.
        let geom_seed_hit = first_iteration
            && pivot_row_hint.is_some_and(|h| !used_rows[h] && r_col[h].norm() > 1e-12);
        if geom_seed_hit {
            let h = pivot_row_hint.unwrap();
            pivot_row = h;
            max_val = r_col[h].norm();
            found = true;
        } else {
            for i in 0..m {
                if used_rows[i] {
                    continue;
                }
                let v = r_col[i].norm();
                if !found || v > max_val {
                    max_val = v;
                    pivot_row = i;
                    found = true;
                }
            }
        }

        if !found {
            // All m rows used — exact reconstruction within row span.
            converged = true;
            break;
        }

        if max_val < 1e-15 {
            // Reference column is dead; try another unused column once before
            // declaring convergence.
            let new_ref = used_cols
                .iter()
                .enumerate()
                .find(|&(jj, &used)| !used && jj != j_ref)
                .map(|(jj, _)| jj);
            match new_ref {
                Some(jj) => {
                    j_ref = jj;
                    r_col = column_residual(j_ref, &u_cols, &v_cols);
                    let mut alt_max = 0.0_f64;
                    let mut alt_pivot = 0_usize;
                    let mut alt_found = false;
                    for i in 0..m {
                        if used_rows[i] {
                            continue;
                        }
                        let v = r_col[i].norm();
                        if !alt_found || v > alt_max {
                            alt_max = v;
                            alt_pivot = i;
                            alt_found = true;
                        }
                    }
                    if !alt_found || alt_max < 1e-15 {
                        converged = true;
                        break;
                    }
                    pivot_row = alt_pivot;
                    max_val = alt_max;
                }
                None => {
                    converged = true;
                    break;
                }
            }
        }
        let _ = max_val; // keep for diagnostics; not used after the pivot is locked

        // Step 2: full residual row at the chosen pivot row.
        let r_row = row_residual(pivot_row, &u_cols, &v_cols);

        // Step 3: column pivot from the row residual, excluding used columns.
        let mut pivot_col = 0_usize;
        let mut col_max = 0.0_f64;
        let mut col_found = false;
        for j in 0..n {
            if used_cols[j] {
                continue;
            }
            let v = r_row[j].norm();
            if !col_found || v > col_max {
                col_max = v;
                pivot_col = j;
                col_found = true;
            }
        }
        if !col_found || col_max < 1e-15 {
            converged = true;
            break;
        }

        let delta = r_row[pivot_col];
        let v_new: Vec<Complex64> = r_row.iter().map(|&x| x / delta).collect();

        // Step 4: column residual at the chosen column pivot — this is the new
        // u factor. We could reuse `r_col` if `pivot_col == j_ref`, but the
        // extra m kernel evals when they differ is what buys ACA+ its
        // robustness, so we always recompute.
        let u_new = if pivot_col == j_ref {
            r_col
        } else {
            column_residual(pivot_col, &u_cols, &v_cols)
        };

        // Frobenius update with full cross-term accounting (mirrors the
        // logic in `aca_partial_pivot` for parity of the convergence test).
        let u_norm_sq: f64 = u_new.iter().map(|x| x.norm_sqr()).sum();
        let v_norm_sq: f64 = v_new.iter().map(|x| x.norm_sqr()).sum();
        let delta_frob = u_norm_sq.sqrt() * v_norm_sq.sqrt();

        // ‖U V^T‖_F² cross-term update — see the matching block in
        // `aca_partial_pivot` for the derivation. The convention switch from
        // U V^H to U V^T (see `HMatrix::matvec`) drops the conjugate on
        // `v_dot`; with `v_dot.conj()` the convergence test is phase-sensitive
        // for complex kernels.
        frobenius_sq += u_norm_sq * v_norm_sq;
        for (u_l, v_l) in u_cols.iter().zip(v_cols.iter()) {
            let u_dot: Complex64 = u_new
                .iter()
                .zip(u_l.iter())
                .map(|(a, b)| a * b.conj())
                .sum();
            let v_dot: Complex64 = v_new
                .iter()
                .zip(v_l.iter())
                .map(|(a, b)| a * b.conj())
                .sum();
            frobenius_sq += 2.0 * (u_dot * v_dot).re;
        }

        u_cols.push(u_new);
        v_cols.push(v_new);
        used_rows[pivot_row] = true;
        used_cols[pivot_col] = true;
        first_iteration = false;

        // Refresh the reference column if we just consumed it.
        if pivot_col == j_ref {
            j_ref = used_cols
                .iter()
                .enumerate()
                .find(|&(_, &used)| !used)
                .map(|(jj, _)| jj)
                .unwrap_or(j_ref);
        }

        let frobenius_norm = frobenius_sq.sqrt();
        if delta_frob < aca_tolerance * frobenius_norm && u_cols.len() > 1 {
            converged = true;
            break;
        }
    }

    let rank = u_cols.len();
    if !converged && !cap_is_user_supplied {
        converged = true;
    }

    let mut u_matrix = DenseMatrix::<Complex64>::zeros(m, rank);
    let mut v_matrix = DenseMatrix::<Complex64>::zeros(n, rank);
    for (k, u_col) in u_cols.iter().enumerate() {
        for i in 0..m {
            u_matrix[[i, k]] = u_col[i];
        }
    }
    for (k, v_col) in v_cols.iter().enumerate() {
        for j in 0..n {
            v_matrix[[j, k]] = v_col[j];
        }
    }

    (u_matrix, v_matrix, rank, converged)
}

// ============================================================================
// H-matrix construction
// ============================================================================

/// Build an H-matrix using HCA for far-field block compression.
///
/// Near-field blocks (clusters that are close together) are stored as dense matrices.
/// Far-field blocks (well-separated clusters) are compressed using HCA by
/// default. To select ACA fallback variants instead, call
/// `build_hmatrix_with_variant`.
///
/// # Arguments
/// * `elements` - All mesh elements
/// * `nodes` - Node coordinates (num_nodes x 3)
/// * `clusters` - Leaf-level clusters from octree decomposition
/// * `physics` - Physics parameters (wave number, frequency, etc.)
/// * `aca_tolerance` - ACA convergence tolerance (typically 1e-4)
/// * `max_rank` - Maximum rank for any low-rank block (typically min(n_src, n_fld) / 2)
///
/// # Returns
/// Assembled H-matrix ready for matrix-vector products
pub fn build_hmatrix(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
) -> HMatrix {
    build_hmatrix_with_variant_and_precision(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        AcaVariant::default(),
        LowRankPrecision::default(),
    )
}

/// Build an H-matrix with an explicit ACA variant. Defaults to `Complex64`
/// far-field storage. For the f32 variant (Phase 1.6, halves far-field
/// memory at the cost of block-relative precision ≈ 1×10⁻⁷), call
/// `build_hmatrix_with_variant_and_precision` directly.
///
/// See `AcaVariant` for the available pivot strategies. `build_hmatrix`
/// delegates here with `AcaVariant::default()` (HCA) and
/// `LowRankPrecision::default()` (`F64`).
pub fn build_hmatrix_with_variant(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    variant: AcaVariant,
) -> HMatrix {
    build_hmatrix_with_variant_and_precision(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        variant,
        LowRankPrecision::default(),
    )
}

fn quantize_morton_coord(value: f64, min: f64, span: f64) -> u32 {
    if span <= f64::EPSILON {
        return 0;
    }
    let scaled = ((value - min) / span).clamp(0.0, 1.0);
    (scaled * ((1_u32 << 21) - 1) as f64).round() as u32
}

fn interleave_morton3(x: u32, y: u32, z: u32) -> u64 {
    let mut code = 0_u64;
    for bit in 0..21 {
        code |= ((x as u64 >> bit) & 1) << (3 * bit);
        code |= ((y as u64 >> bit) & 1) << (3 * bit + 1);
        code |= ((z as u64 >> bit) & 1) << (3 * bit + 2);
    }
    code
}

fn morton_ordered_cluster_indices(clusters: &[Cluster]) -> Vec<usize> {
    if clusters.is_empty() {
        return Vec::new();
    }

    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for cluster in clusters {
        for axis in 0..3 {
            min[axis] = min[axis].min(cluster.center[axis]);
            max[axis] = max[axis].max(cluster.center[axis]);
        }
    }
    let span = [
        (max[0] - min[0]).max(f64::EPSILON),
        (max[1] - min[1]).max(f64::EPSILON),
        (max[2] - min[2]).max(f64::EPSILON),
    ];

    let mut keyed: Vec<(u64, usize)> = clusters
        .iter()
        .enumerate()
        .map(|(idx, cluster)| {
            let x = quantize_morton_coord(cluster.center[0], min[0], span[0]);
            let y = quantize_morton_coord(cluster.center[1], min[1], span[1]);
            let z = quantize_morton_coord(cluster.center[2], min[2], span[2]);
            (interleave_morton3(x, y, z), idx)
        })
        .collect();
    keyed.sort_unstable_by_key(|&(key, idx)| (key, idx));
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

fn default_hmatrix_sign(physics: &PhysicsParams) -> f64 {
    // The boundary-integral formulation sign is a convention, not a
    // frequency-dependent tuning parameter. Keep the convenience builders in
    // lockstep with BemSolver and the explicit stabilized builders.
    PhysicsParams::optimal_formulation_sign(physics.wave_number)
}

/// Build an H-matrix with explicit control over both the ACA variant and
/// the stored precision of the low-rank `(U, V)` factors. ACA itself
/// always runs in `Complex64`; `precision == LowRankPrecision::F32` only
/// affects the post-build storage and the matvec arithmetic intensity.
pub fn build_hmatrix_with_variant_and_precision(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    variant: AcaVariant,
    precision: LowRankPrecision,
) -> HMatrix {
    let sign = default_hmatrix_sign(physics);
    let beta = physics.burton_miller_beta();
    build_hmatrix_with_variant_precision_and_beta(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        variant,
        precision,
        sign,
        beta,
    )
}

/// Build an H-matrix with the same stabilised Burton-Miller parameters used by
/// `tbem::build_tbem_system_stabilized`.
pub fn build_hmatrix_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    sign: f64,
    beta_scale: f64,
) -> HMatrix {
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    build_hmatrix_with_variant_precision_and_beta(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        AcaVariant::default(),
        LowRankPrecision::default(),
        sign,
        beta,
    )
}

/// Build a compressed H-matrix for the fixed-geometry wavenumber derivative
/// of the stabilised TBEM operator.
///
/// The derivative uses the same near/far block topology, ACA variant, rank
/// cap, and low-rank storage precision as [`build_hmatrix_stabilized`]. Near
/// blocks use the complete singular/curved entry derivative; far blocks use
/// analytic entry derivatives during ACA, so the result remains compressed
/// instead of falling back to the full-rank derivative reference oracle.
/// Build a compressed H-matrix fixed-geometry density derivative.
///
/// The derivative shares the value operator's ACA topology and evaluates
/// the explicit impedance pressure-factor dependence analytically.
#[allow(clippy::too_many_arguments)]
pub fn build_hmatrix_stabilized_density_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    variant: AcaVariant,
    precision: LowRankPrecision,
    sign: f64,
    beta_scale: f64,
) -> Result<HMatrix, MatrixDerivativeError> {
    if !sign.is_finite() || !beta_scale.is_finite() {
        return Err(MatrixDerivativeError::InvalidParameter {
            name: "sign and beta scale",
        });
    }
    let mode = MatrixEntryMode::DensityDerivative {
        gamma: Complex64::new(physics.gamma(), 0.0),
        tau: Complex64::new(physics.tau, 0.0),
        beta: physics.burton_miller_beta_scaled(beta_scale),
        sign,
    };
    mode.validate()?;
    Ok(build_hmatrix_with_variant_precision_and_beta_mode(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        variant,
        precision,
        mode,
    ))
}

/// Build the default compressed H-matrix fixed-geometry density derivative.
pub fn build_hmatrix_density_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
) -> Result<HMatrix, MatrixDerivativeError> {
    build_hmatrix_stabilized_density_derivative(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        AcaVariant::default(),
        LowRankPrecision::default(),
        default_hmatrix_sign(physics),
        1.0,
    )
}

/// Build a compressed H-matrix fixed-geometry wavenumber derivative
/// stabilised TBEM operator, with explicit ACA variant, storage precision,
/// sign, beta scale, and regular-quadrature order.
#[allow(clippy::too_many_arguments)]
pub fn build_hmatrix_stabilized_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    variant: AcaVariant,
    precision: LowRankPrecision,
    sign: f64,
    beta_scale: f64,
    gauss_order: usize,
) -> Result<HMatrix, MatrixDerivativeError> {
    if !sign.is_finite() || !beta_scale.is_finite() {
        return Err(MatrixDerivativeError::InvalidParameter {
            name: "sign and beta scale",
        });
    }
    let mode = MatrixEntryMode::WavenumberDerivative {
        gamma: Complex64::new(physics.gamma(), 0.0),
        tau: Complex64::new(physics.tau, 0.0),
        beta: physics.burton_miller_beta_scaled(beta_scale),
        beta_derivative: physics.burton_miller_beta_wavenumber_derivative() * beta_scale,
        sign,
        gauss_order,
    };
    mode.validate()?;
    Ok(build_hmatrix_with_variant_precision_and_beta_mode(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        variant,
        precision,
        mode,
    ))
}

/// Build the default compressed H-matrix for the fixed-geometry wavenumber
/// derivative using the default HCA variant and `Complex64` factors.
pub fn build_hmatrix_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    gauss_order: usize,
) -> Result<HMatrix, MatrixDerivativeError> {
    build_hmatrix_stabilized_wavenumber_derivative(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        AcaVariant::default(),
        LowRankPrecision::default(),
        default_hmatrix_sign(physics),
        1.0,
        gauss_order,
    )
}

/// Build an H-matrix with explicit stabilized Burton--Miller parameters.
///
/// This is the shared constructor used by dense-parity wrappers that must
/// use exactly the same `sign` and `beta` as a TBEM reference assembly.
pub fn build_hmatrix_with_variant_precision_and_beta(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    variant: AcaVariant,
    precision: LowRankPrecision,
    sign: f64,
    beta: Complex64,
) -> HMatrix {
    build_hmatrix_with_variant_precision_and_beta_mode(
        elements,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        variant,
        precision,
        MatrixEntryMode::Value {
            gamma: Complex64::new(physics.gamma(), 0.0),
            tau: Complex64::new(physics.tau, 0.0),
            beta,
            sign,
        },
    )
}

/// Build one proxy element per panel-local higher-order DOF.
///
/// The ACA implementation is deliberately shared with the constant-panel
/// path. Proxies carry the original geometry and boundary condition, while
/// their center is the source collocation point and their group field encodes
/// the local basis slot. The resulting factors therefore have exactly the
/// dimensions expected by `cluster_dof_indices` and H-matrix matvecs.
fn build_dof_proxies(elements: &[Element], nodes: &[[f64; 3]]) -> (Vec<Element>, Vec<usize>) {
    let max_dof = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .flat_map(|element| element.dof_addresses.iter().copied())
        .max()
        .map(|dof| dof + 1)
        .unwrap_or(0);
    let mut proxies = Vec::new();
    let mut dof_to_proxy = vec![usize::MAX; max_dof];
    for element in elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
    {
        let Some(basis) = basis_order_for_dof_count(element, element.dof_addresses.len()) else {
            continue;
        };
        let collocation_points = element.collocation_points(basis, nodes);
        let collocation_normals = element.collocation_normals(basis, nodes);
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            if dof >= dof_to_proxy.len() || dof_to_proxy[dof] != usize::MAX {
                continue;
            }
            let mut proxy = element.clone();
            proxy.center = collocation_points
                .get(local)
                .copied()
                .unwrap_or(element.center);
            proxy.normal = collocation_normals
                .get(local)
                .copied()
                .unwrap_or(element.normal);
            proxy.dof_addresses = vec![dof];
            proxy.group = encode_dof_proxy_group(element.dof_addresses.len(), local);
            dof_to_proxy[dof] = proxies.len();
            proxies.push(proxy);
        }
    }
    (proxies, dof_to_proxy)
}

#[allow(clippy::too_many_arguments)]
fn build_hmatrix_with_variant_precision_and_beta_mode(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    variant: AcaVariant,
    precision: LowRankPrecision,
    entry_mode: MatrixEntryMode,
) -> HMatrix {
    // Ordinary H-matrix inputs are either discontinuous panel DOFs or the
    // one-proxy-per-DOF layout produced by `build_dof_proxies`, so summing
    // local addresses gives the compact operator dimension.  Continuous
    // higher-order proxy layouts intentionally retain repeated global
    // addresses for every panel incidence; use the global address space in
    // that case while keeping the repeated cluster entries for matvec
    // accumulation.
    let expanded_shared_proxies = elements
        .iter()
        .any(|element| has_shared_dof_proxy_group(element.group));
    let num_dofs = if expanded_shared_proxies {
        elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .flat_map(|element| element.dof_addresses.iter().copied())
            .max()
            .map_or(0, |max_dof| max_dof + 1)
    } else {
        elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .map(|element| element.dof_addresses.len())
            .sum()
    };

    // O(N) precomputation of `dof → element_index` so the dense-block fill
    // loop can resolve owners in O(1) instead of scanning every element on
    // every entry. Without this, dense assembly is O(N³ / L) for K clusters
    // of mean size L and dominates wall-clock at moderate sizes (~5k DOFs).
    let dof_to_element: Vec<usize> = {
        let max_dof = elements
            .iter()
            .filter(|e| !e.property.is_evaluation())
            .flat_map(|e| e.dof_addresses.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let mut table = vec![usize::MAX; max_dof];
        for (idx, elem) in elements.iter().enumerate() {
            if elem.property.is_evaluation() {
                continue;
            }
            for &dof in &elem.dof_addresses {
                if dof < max_dof {
                    table[dof] = idx;
                }
            }
        }
        table
    };

    // Build cluster DOF index mappings
    let cluster_dof_indices: Vec<Vec<usize>> = clusters
        .iter()
        .map(|cluster| {
            cluster
                .element_indices
                .iter()
                .filter(|&&ei| !elements[ei].property.is_evaluation())
                .flat_map(|&ei| elements[ei].dof_addresses.clone())
                .collect()
        })
        .collect();
    let cluster_element_indices: Vec<Vec<usize>> = clusters
        .iter()
        .map(|cluster| {
            cluster
                .element_indices
                .iter()
                .filter(|&&element_index| !elements[element_index].property.is_evaluation())
                .copied()
                .collect()
        })
        .collect();
    // Both value and analytic-derivative ACA need one proxy per local
    // higher-order DOF. Restricting this to the value mode leaves derivative
    // factors sized by element count while matvec gathers global DOFs.
    let higher_order_mode = !expanded_shared_proxies
        && elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .any(|element| element.dof_addresses.len() > 1);
    let (dof_proxies, dof_to_proxy) = if higher_order_mode {
        let (proxies, map) = build_dof_proxies(elements, nodes);
        (Some(proxies), Some(map))
    } else {
        (None, None)
    };
    let source_order = morton_ordered_cluster_indices(clusters);

    struct BuiltBlocks {
        source_cluster: usize,
        dense_blocks: Vec<DenseBlock>,
        low_rank_blocks: Vec<LowRankBlock>,
        memory_bytes: usize,
    }

    // Cluster block construction (rayon above the machine-adaptive gate,
    // serial for few clusters). Each cluster build runs ACA/dense kernels,
    // so the grain is coarse blocks per thread.
    let built_blocks: Vec<BuiltBlocks> = gated_map(
        source_order.as_slice(),
        source_order.len(),
        BEM_BLOCKS_PER_THREAD,
        |&src_idx| {
            let src_cluster = &clusters[src_idx];
            let mut dense_blocks = Vec::new();
            let mut low_rank_blocks = Vec::new();
            let mut memory_bytes = 0_usize;

            // Near-field: dense blocks
            for &fld_idx in &src_cluster.near_clusters {
                let src_dofs = &cluster_dof_indices[src_idx];
                let fld_dofs = &cluster_dof_indices[fld_idx];

                if src_dofs.is_empty() || fld_dofs.is_empty() {
                    continue;
                }

                let m = src_dofs.len();
                let n = fld_dofs.len();
                let block_matrix = assemble_dense_block_matrix_with_mode(
                    src_dofs,
                    fld_dofs,
                    &cluster_element_indices[src_idx],
                    &cluster_element_indices[fld_idx],
                    &dof_to_element,
                    elements,
                    nodes,
                    physics,
                    entry_mode,
                );

                memory_bytes += m * n * 16; // 16 bytes per Complex64
                dense_blocks.push(DenseBlock {
                    source_cluster: src_idx,
                    field_cluster: fld_idx,
                    matrix: block_matrix,
                });
            }

            // Far-field: ACA compressed blocks
            for &fld_idx in &src_cluster.far_clusters {
                let (src_indices, fld_indices, aca_elements): (Vec<usize>, Vec<usize>, &[Element]) =
                    if higher_order_mode {
                        let proxy_map = dof_to_proxy
                            .as_ref()
                            .expect("higher-order H-matrix proxy map missing");
                        let proxies = dof_proxies
                            .as_ref()
                            .expect("higher-order H-matrix proxies missing");
                        let src = cluster_dof_indices[src_idx]
                            .iter()
                            .filter_map(|&dof| proxy_map.get(dof).copied())
                            .filter(|&proxy| proxy != usize::MAX)
                            .collect();
                        let fld = cluster_dof_indices[fld_idx]
                            .iter()
                            .filter_map(|&dof| proxy_map.get(dof).copied())
                            .filter(|&proxy| proxy != usize::MAX)
                            .collect();
                        (src, fld, proxies.as_slice())
                    } else {
                        let src = src_cluster
                            .element_indices
                            .iter()
                            .filter(|&&ei| !elements[ei].property.is_evaluation())
                            .copied()
                            .collect();
                        let fld = clusters[fld_idx]
                            .element_indices
                            .iter()
                            .filter(|&&ei| !elements[ei].property.is_evaluation())
                            .copied()
                            .collect();
                        (src, fld, elements)
                    };

                if src_indices.is_empty() || fld_indices.is_empty() {
                    continue;
                }

                let m = src_indices.len();
                let n = fld_indices.len();
                let block_max_rank = max_rank.min(m / 2 + 1).min(n / 2 + 1);

                let (u, v, rank, converged) = match variant {
                    AcaVariant::PartialPivot => aca_partial_pivot_with_mode(
                        &src_indices,
                        &fld_indices,
                        aca_elements,
                        nodes,
                        physics,
                        entry_mode,
                        aca_tolerance,
                        block_max_rank,
                    ),
                    AcaVariant::Plus => aca_plus_with_mode(
                        &src_indices,
                        &fld_indices,
                        aca_elements,
                        nodes,
                        physics,
                        entry_mode,
                        aca_tolerance,
                        block_max_rank,
                    ),
                    AcaVariant::Gp => aca_gp_with_mode(
                        &src_indices,
                        &fld_indices,
                        aca_elements,
                        nodes,
                        physics,
                        entry_mode,
                        aca_tolerance,
                        block_max_rank,
                    ),
                    AcaVariant::Hca => hca_with_mode(
                        &src_indices,
                        &fld_indices,
                        aca_elements,
                        nodes,
                        physics,
                        entry_mode,
                        aca_tolerance,
                        block_max_rank,
                    ),
                };

                if !converged {
                    let src_dofs = &cluster_dof_indices[src_idx];
                    let fld_dofs = &cluster_dof_indices[fld_idx];
                    let block_matrix = assemble_dense_block_matrix_with_mode(
                        src_dofs,
                        fld_dofs,
                        &cluster_element_indices[src_idx],
                        &cluster_element_indices[fld_idx],
                        &dof_to_element,
                        elements,
                        nodes,
                        physics,
                        entry_mode,
                    );
                    memory_bytes += src_dofs.len() * fld_dofs.len() * 16;
                    dense_blocks.push(DenseBlock {
                        source_cluster: src_idx,
                        field_cluster: fld_idx,
                        matrix: block_matrix,
                    });
                } else if rank > 0 {
                    let factors = match precision {
                        LowRankPrecision::F64 => LowRankFactors::F64 { u, v },
                        LowRankPrecision::F32 => downcast_factors_f64_to_f32(u, v),
                    };
                    memory_bytes += factors.bytes();
                    low_rank_blocks.push(LowRankBlock {
                        source_cluster: src_idx,
                        field_cluster: fld_idx,
                        factors,
                        rank,
                        requested_max_rank: block_max_rank,
                        converged,
                    });
                }
            }

            BuiltBlocks {
                source_cluster: src_idx,
                dense_blocks,
                low_rank_blocks,
                memory_bytes,
            }
        },
    );

    let mut dense_blocks = Vec::new();
    let mut low_rank_blocks = Vec::new();
    let mut partitions = Vec::new();
    let mut total_memory = 0_usize;
    let num_partitions = rayon::current_num_threads()
        .max(1)
        .min(built_blocks.len().max(1));
    let partition_len = built_blocks.len().max(1).div_ceil(num_partitions);
    let mut active_partition: Option<HMatrixPartition> = None;
    for (idx, blocks) in built_blocks.into_iter().enumerate() {
        if idx % partition_len.max(1) == 0 {
            if let Some(mut partition) = active_partition.take() {
                partition.dense_end = dense_blocks.len();
                partition.low_rank_end = low_rank_blocks.len();
                partitions.push(partition);
            }
            active_partition = Some(HMatrixPartition {
                cluster_indices: Vec::new(),
                dense_start: dense_blocks.len(),
                dense_end: dense_blocks.len(),
                low_rank_start: low_rank_blocks.len(),
                low_rank_end: low_rank_blocks.len(),
                num_dofs: 0,
            });
        }

        total_memory += blocks.memory_bytes;
        if let Some(partition) = active_partition.as_mut() {
            partition.cluster_indices.push(blocks.source_cluster);
            partition.num_dofs += cluster_dof_indices[blocks.source_cluster].len();
        }
        dense_blocks.extend(blocks.dense_blocks);
        low_rank_blocks.extend(blocks.low_rank_blocks);
    }
    if let Some(mut partition) = active_partition {
        partition.dense_end = dense_blocks.len();
        partition.low_rank_end = low_rank_blocks.len();
        partitions.push(partition);
    }

    // Field-cluster (transpose) schedule. Blocks are stored in row order,
    // so transpose partitions index through per-kind order arrays sorted by
    // the Morton position of each block's field cluster. Cluster chunking
    // matches the row schedule (`partition_len` clusters per partition) so
    // forward and transpose matvecs use the same number of partitions.
    let (transpose_dense_order, transpose_low_rank_order, transpose_partitions) = {
        let num_clusters = clusters.len();
        let mut morton_pos = vec![usize::MAX; num_clusters];
        for (pos, &cluster_idx) in source_order.iter().enumerate() {
            morton_pos[cluster_idx] = pos;
        }
        let mut dense_order: Vec<usize> = (0..dense_blocks.len()).collect();
        dense_order.sort_by_key(|&b| morton_pos[dense_blocks[b].field_cluster]);
        let mut low_rank_order: Vec<usize> = (0..low_rank_blocks.len()).collect();
        low_rank_order.sort_by_key(|&b| morton_pos[low_rank_blocks[b].field_cluster]);

        let mut dense_by_field: Vec<usize> = vec![0; num_clusters];
        for block in &dense_blocks {
            dense_by_field[block.field_cluster] += 1;
        }
        let mut low_rank_by_field: Vec<usize> = vec![0; num_clusters];
        for block in &low_rank_blocks {
            low_rank_by_field[block.field_cluster] += 1;
        }

        let mut t_partitions = Vec::new();
        let mut dense_cursor = 0_usize;
        let mut low_rank_cursor = 0_usize;
        for chunk in source_order.chunks(partition_len.max(1)) {
            let dense_count: usize = chunk.iter().map(|&c| dense_by_field[c]).sum();
            let low_rank_count: usize = chunk.iter().map(|&c| low_rank_by_field[c]).sum();
            let num_dofs: usize = chunk.iter().map(|&c| cluster_dof_indices[c].len()).sum();
            t_partitions.push(HMatrixTransposePartition {
                cluster_indices: chunk.to_vec(),
                dense_start: dense_cursor,
                dense_end: dense_cursor + dense_count,
                low_rank_start: low_rank_cursor,
                low_rank_end: low_rank_cursor + low_rank_count,
                num_dofs,
            });
            dense_cursor += dense_count;
            low_rank_cursor += low_rank_count;
        }
        (dense_order, low_rank_order, t_partitions)
    };

    let dense_memory = num_dofs * num_dofs * 16;
    let compression_ratio = if total_memory > 0 {
        dense_memory as f64 / total_memory as f64
    } else {
        1.0
    };

    HMatrix {
        num_dofs,
        dense_blocks,
        low_rank_blocks,
        cluster_dof_indices,
        partitions,
        transpose_dense_order,
        transpose_low_rank_order,
        transpose_partitions,
        memory_bytes: total_memory,
        compression_ratio,
    }
}

// ============================================================================
// H-matrix vector product (LinearOperator implementation)
// ============================================================================

/// Per-thread matvec scratch: gather/scatter buffers reused across blocks
/// and across matvec calls. Follows the `SlfmmMatvecWorkspace` precedent in
/// `core/solver/batched_blas.rs`, but thread-local because the H-matrix
/// matvec is invoked through `&self` from solver kernels that do not thread
/// a workspace through.
///
/// Steady-state a matvec allocates nothing in the block loop; the forward
/// row-partition path still uses its existing per-partition update vectors,
/// while the transpose/Hermitian path scatters directly into disjoint output
/// partitions. Scratch buffers may grow on the first call(s) against larger
/// blocks.
//
// The adjoint scatter path uses a raw output pointer because its partition
// schedule proves disjoint writes but Rust cannot express that non-contiguous
// ownership as independent mutable slices. Keep the Send/Sync wrapper local
// to this module and only construct it after the output buffer is validated.
#[derive(Clone, Copy)]
struct ScatterPtr(*mut Complex64);

// SAFETY: callers must uphold the documented disjoint-partition write
// invariant. The only caller is `matvec_adjoint_like_into`, which does so.
unsafe impl Send for ScatterPtr {}
unsafe impl Sync for ScatterPtr {}

impl ScatterPtr {
    /// SAFETY: the caller must own the indexed output location exclusively.
    unsafe fn add(self, index: usize) -> *mut Complex64 {
        unsafe { self.0.add(index) }
    }
}

#[derive(Default)]
struct HmatvecScratch {
    /// Gathered block-local input slice.
    x_local: Vec<Complex64>,
    /// Block-local output slice.
    y_local: Vec<Complex64>,
    /// Rank-sized scratch for the low-rank two-stage kernels.
    t: Vec<Complex64>,
}

impl HmatvecScratch {
    fn gather(&mut self, x: &[Complex64], dofs: &[usize]) {
        self.x_local.clear();
        self.x_local.extend(dofs.iter().map(|&j| x[j]));
    }

    fn zero_y(&mut self, len: usize) {
        self.y_local.clear();
        self.y_local.resize(len, Complex64::new(0.0, 0.0));
    }
}

thread_local! {
    static HMATVEC_SCRATCH: RefCell<HmatvecScratch> = RefCell::new(HmatvecScratch::default());
}

impl HMatrix {
    /// Upload this compressed operator to a reusable CUDA Complex64
    /// representation.
    ///
    /// Dense near-field blocks and both factor orientations (including their
    /// conjugated Hermitian forms) are uploaded once.  Subsequent actions on
    /// the returned object keep vectors, index maps, factors, and projection
    /// scratch on the device.  The supplied pool owns the allocation quota.
    #[cfg(feature = "cuda")]
    pub fn try_to_cuda(
        &self,
        pool: MemoryPool<CudaBackend>,
    ) -> sonium_compute::BackendResult<CudaHMatrixC64> {
        let dense_blocks = self
            .dense_blocks
            .iter()
            .enumerate()
            .map(|(block, value)| {
                let row_indices = self
                    .cluster_dof_indices
                    .get(value.source_cluster)
                    .cloned()
                    .ok_or_else(|| sonium_compute::BackendError::InvalidArgument {
                        operation: "hmatrix_to_cuda",
                        reason: format!("dense block {block} has invalid source cluster"),
                    })?;
                let col_indices = self
                    .cluster_dof_indices
                    .get(value.field_cluster)
                    .cloned()
                    .ok_or_else(|| sonium_compute::BackendError::InvalidArgument {
                        operation: "hmatrix_to_cuda",
                        reason: format!("dense block {block} has invalid field cluster"),
                    })?;
                Ok(CudaDenseBlockC64Input {
                    row_indices,
                    col_indices,
                    matrix: value.matrix.as_slice().to_vec(),
                })
            })
            .collect::<sonium_compute::BackendResult<Vec<_>>>()?;

        let low_rank_blocks = self
            .low_rank_blocks
            .iter()
            .enumerate()
            .map(|(block, value)| {
                let row_indices = self
                    .cluster_dof_indices
                    .get(value.source_cluster)
                    .cloned()
                    .ok_or_else(|| sonium_compute::BackendError::InvalidArgument {
                        operation: "hmatrix_to_cuda",
                        reason: format!("low-rank block {block} has invalid source cluster"),
                    })?;
                let col_indices = self
                    .cluster_dof_indices
                    .get(value.field_cluster)
                    .cloned()
                    .ok_or_else(|| sonium_compute::BackendError::InvalidArgument {
                        operation: "hmatrix_to_cuda",
                        reason: format!("low-rank block {block} has invalid field cluster"),
                    })?;
                let (u, v) = match &value.factors {
                    LowRankFactors::F64 { u, v } => (u.as_slice().to_vec(), v.as_slice().to_vec()),
                    LowRankFactors::F32 {
                        u,
                        v,
                        scale_u,
                        scale_v,
                    } => (
                        u.as_slice()
                            .iter()
                            .map(|value| {
                                Complex64::new(
                                    value.re as f64 * *scale_u,
                                    value.im as f64 * *scale_u,
                                )
                            })
                            .collect(),
                        v.as_slice()
                            .iter()
                            .map(|value| {
                                Complex64::new(
                                    value.re as f64 * *scale_v,
                                    value.im as f64 * *scale_v,
                                )
                            })
                            .collect(),
                    ),
                };
                Ok(CudaLowRankBlockC64Input {
                    row_indices,
                    col_indices,
                    u,
                    v,
                    rank: value.rank,
                })
            })
            .collect::<sonium_compute::BackendResult<Vec<_>>>()?;

        CudaHMatrixC64::new(pool, self.num_dofs, dense_blocks, low_rank_blocks)
    }

    /// Upload a production H-matrix and its fixed-geometry wavenumber
    /// derivative as one resident first-order update operator.
    ///
    /// The two host-built operators must describe the same global layout;
    /// subsequent `try_apply` calls combine their actions on the device and
    /// validate the caller's base-wavenumber metadata before dispatch.
    #[cfg(feature = "cuda")]
    pub fn try_to_cuda_wavenumber_update(
        &self,
        derivative: &Self,
        pool: MemoryPool<CudaBackend>,
        base_wavenumber: f64,
    ) -> sonium_compute::BackendResult<CudaHMatrixWavenumberUpdateC64> {
        if self.num_dofs != derivative.num_dofs {
            return Err(sonium_compute::BackendError::InvalidArgument {
                operation: "hmatrix_to_cuda_wavenumber_update",
                reason: format!(
                    "value and derivative dimensions differ ({} vs {})",
                    self.num_dofs, derivative.num_dofs
                ),
            });
        }
        let value = self.try_to_cuda(pool.clone())?;
        let derivative = derivative.try_to_cuda(pool.clone())?;
        CudaHMatrixWavenumberUpdateC64::new(value, derivative, pool, base_wavenumber)
    }

    /// Summarise the realised low-rank ranks against their per-block caps.
    ///
    /// Returns an empty (all-zero) `RankStats` when there are no low-rank
    /// blocks (e.g. small problems where every cluster pair is near-field).
    pub fn rank_stats(&self) -> RankStats {
        let block_count = self.low_rank_blocks.len();
        if block_count == 0 {
            return RankStats {
                block_count: 0,
                min_rank: 0,
                max_rank: 0,
                mean_rank: 0.0,
                p50_rank: 0,
                p90_rank: 0,
                p99_rank: 0,
                mean_requested_max_rank: 0.0,
                saturated_count: 0,
                saturation_fraction: 0.0,
            };
        }

        let mut ranks: Vec<usize> = self.low_rank_blocks.iter().map(|b| b.rank).collect();
        ranks.sort_unstable();

        let sum_rank: usize = ranks.iter().sum();
        let mean_rank = sum_rank as f64 / block_count as f64;
        let mean_requested_max_rank = self
            .low_rank_blocks
            .iter()
            .map(|b| b.requested_max_rank)
            .sum::<usize>() as f64
            / block_count as f64;

        // Nearest-rank percentile on sorted ranks.
        let pct = |p: f64| -> usize {
            let idx = ((p * block_count as f64).ceil() as usize).saturating_sub(1);
            ranks[idx.min(block_count - 1)]
        };

        let saturated_count = self.low_rank_blocks.iter().filter(|b| !b.converged).count();

        RankStats {
            block_count,
            min_rank: ranks[0],
            max_rank: ranks[block_count - 1],
            mean_rank,
            p50_rank: pct(0.50),
            p90_rank: pct(0.90),
            p99_rank: pct(0.99),
            mean_requested_max_rank,
            saturated_count,
            saturation_fraction: saturated_count as f64 / block_count as f64,
        }
    }

    /// Build a reference derivative operator with the same block topology.
    ///
    /// Dense blocks are differentiated entry-by-entry. Far-field blocks are
    /// represented as full-rank `U · Iᵀ` factors, so this method preserves the
    /// H-matrix scheduling and transpose contracts while deliberately
    /// avoiding an approximation of the derivative itself. It is an
    /// O(N²)-storage validation oracle for a future compressed derivative
    /// builder, not a production memory-saving path.
    pub fn build_wavenumber_derivative_reference(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        sign: f64,
        beta_scale: f64,
        gauss_order: usize,
    ) -> Result<Self, MatrixDerivativeError> {
        if gauss_order == 0 {
            return Err(MatrixDerivativeError::InvalidQuadratureOrder);
        }
        if !sign.is_finite() || !beta_scale.is_finite() {
            return Err(MatrixDerivativeError::InvalidParameter {
                name: "sign and beta_scale",
            });
        }

        let dof_to_element = derivative_dof_owner_table(elements);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta_scaled(beta_scale);
        let beta_derivative = physics.burton_miller_beta_wavenumber_derivative() * beta_scale;

        let dense_blocks: Vec<DenseBlock> = self
            .dense_blocks
            .iter()
            .map(|block| {
                Ok(DenseBlock {
                    source_cluster: block.source_cluster,
                    field_cluster: block.field_cluster,
                    matrix: assemble_wavenumber_derivative_block(
                        &self.cluster_dof_indices[block.source_cluster],
                        &self.cluster_dof_indices[block.field_cluster],
                        &dof_to_element,
                        elements,
                        nodes,
                        physics,
                        gamma,
                        tau,
                        beta,
                        beta_derivative,
                        sign,
                        gauss_order,
                    )?,
                })
            })
            .collect::<Result<_, MatrixDerivativeError>>()?;

        let low_rank_blocks: Vec<LowRankBlock> = self
            .low_rank_blocks
            .iter()
            .map(|block| {
                let source_dofs = &self.cluster_dof_indices[block.source_cluster];
                let field_dofs = &self.cluster_dof_indices[block.field_cluster];
                let derivative = assemble_wavenumber_derivative_block(
                    source_dofs,
                    field_dofs,
                    &dof_to_element,
                    elements,
                    nodes,
                    physics,
                    gamma,
                    tau,
                    beta,
                    beta_derivative,
                    sign,
                    gauss_order,
                )?;
                let mut identity =
                    DenseMatrix::<Complex64>::zeros(field_dofs.len(), field_dofs.len());
                for diagonal in 0..field_dofs.len() {
                    identity[[diagonal, diagonal]] = Complex64::new(1.0, 0.0);
                }
                let rank = field_dofs.len();
                Ok(LowRankBlock {
                    source_cluster: block.source_cluster,
                    field_cluster: block.field_cluster,
                    factors: LowRankFactors::F64 {
                        u: derivative,
                        v: identity,
                    },
                    rank,
                    requested_max_rank: rank,
                    converged: true,
                })
            })
            .collect::<Result<_, MatrixDerivativeError>>()?;

        let memory_bytes = dense_blocks
            .iter()
            .map(|block| block.matrix.nrows() * block.matrix.ncols() * 16)
            .sum::<usize>()
            + low_rank_blocks
                .iter()
                .map(|block| block.factors.bytes())
                .sum::<usize>();
        let dense_bytes = self
            .num_dofs
            .saturating_mul(self.num_dofs)
            .saturating_mul(16);
        let compression_ratio = if memory_bytes == 0 {
            1.0
        } else {
            dense_bytes as f64 / memory_bytes as f64
        };

        Ok(Self {
            num_dofs: self.num_dofs,
            dense_blocks,
            low_rank_blocks,
            cluster_dof_indices: self.cluster_dof_indices.clone(),
            partitions: self.partitions.clone(),
            transpose_dense_order: self.transpose_dense_order.clone(),
            transpose_low_rank_order: self.transpose_low_rank_order.clone(),
            transpose_partitions: self.transpose_partitions.clone(),
            memory_bytes,
            compression_ratio,
        })
    }

    /// H-matrix vector product: y = H * x
    ///
    /// Dense blocks: standard matrix-vector product
    /// Low-rank blocks: y += U * (V^T * x) -- O(k*n) per block instead of O(n^2)
    ///
    /// Convention: ACA construction stores `(U, V)` such that
    /// `A[i, j] = Σ_l U[i, l] * V[j, l]` (no conjugate). This matches the
    /// residual update `R[i, j] -= u_l[i] * v_l[j]` in the ACA loop.
    /// Applying `V^H` here would silently break Hermitian-asymmetric
    /// reconstruction on complex kernels — see
    /// `test_hmatrix_matvec_dense_parity_complex`.
    pub fn matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_into(x, &mut y);
        y
    }

    /// Checked H-matrix product for solver and coupling boundaries.
    pub fn matvec_checked(&self, x: &[Complex64]) -> Result<Vec<Complex64>, HMatrixOperatorError> {
        self.validate_operator_input(x)?;
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_into_checked(x, &mut y)?;
        Ok(y)
    }

    /// Checked allocation-free H-matrix product.
    pub fn matvec_into_checked(
        &self,
        x: &[Complex64],
        y: &mut [Complex64],
    ) -> Result<(), HMatrixOperatorError> {
        self.validate_operator_input(x)?;
        if y.len() != self.num_dofs {
            return Err(HMatrixOperatorError::OutputLength {
                expected: self.num_dofs,
                actual: y.len(),
            });
        }
        self.matvec_into(x, y);
        self.validate_operator_output(y)
    }

    /// Compute the matrix-vector product y = A*x into a pre-allocated buffer.
    pub fn matvec_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.num_dofs);
        y.fill(Complex64::new(0.0, 0.0));

        // Partition matvec (rayon above the machine-adaptive gate, serial
        // for small systems). Work is output entries.
        let partition_updates: Vec<Vec<(usize, Complex64)>> = gated_map(
            &self.partitions,
            self.num_dofs,
            BEM_ENTRIES_PER_THREAD,
            |partition| {
                let mut updates = Vec::with_capacity(partition.num_dofs);

                HMATVEC_SCRATCH.with(|cell| {
                    let scratch = &mut *cell.borrow_mut();

                    // Dense block contributions
                    for block in &self.dense_blocks[partition.dense_start..partition.dense_end] {
                        let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                        let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

                        scratch.gather(x, fld_dofs);
                        scratch.zero_y(src_dofs.len());
                        block
                            .matrix
                            .matvec_into(&scratch.x_local, &mut scratch.y_local);

                        for (i_local, &i_global) in src_dofs.iter().enumerate() {
                            updates.push((i_global, scratch.y_local[i_local]));
                        }
                    }

                    // Low-rank block contributions: y += U * (V^T * x)
                    for block in
                        &self.low_rank_blocks[partition.low_rank_start..partition.low_rank_end]
                    {
                        let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                        let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

                        scratch.gather(x, fld_dofs);
                        scratch.zero_y(src_dofs.len());
                        block.factors.apply_into(
                            &scratch.x_local,
                            &mut scratch.y_local,
                            &mut scratch.t,
                        );

                        for (i_local, &i_global) in src_dofs.iter().enumerate() {
                            updates.push((i_global, scratch.y_local[i_local]));
                        }
                    }
                });

                updates
            },
        );

        for updates in partition_updates {
            for (i_global, value) in updates {
                y[i_global] += value;
            }
        }
    }

    /// Compute `Y = H · X` for multiple right-hand sides using block kernels.
    pub fn matvec_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_block_into(x, &mut y);
        y
    }

    /// Compute `Y = H · X` into a pre-allocated row-major dense matrix.
    ///
    /// Forward partitions own disjoint source-cluster DOF sets, so each
    /// worker can scatter all right-hand sides directly into `y`. Dense
    /// near-field blocks use matrix-matrix multiplication and far-field
    /// blocks use [`LowRankFactors::apply_block_into`], avoiding the trait's
    /// default per-column loop.
    pub fn matvec_block_into(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        assert_eq!(x.nrows(), self.num_dofs);
        assert_eq!(y.nrows(), self.num_dofs);
        assert_eq!(y.ncols(), x.ncols());
        y.fill(Complex64::new(0.0, 0.0));

        let nrhs = x.ncols();
        let y_ptr = ScatterPtr(y.as_mut_slice().as_mut_ptr());
        // Block scatter (rayon above the machine-adaptive gate, serial for
        // small systems). The partition closure is shared by both paths.
        // Work is output entries × right-hand sides.
        let scatter_partition = |partition: &HMatrixPartition| {
            for block in &self.dense_blocks[partition.dense_start..partition.dense_end] {
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];
                let x_local =
                    DenseMatrix::from_fn(fld_dofs.len(), nrhs, |j, rhs| x[[fld_dofs[j], rhs]]);
                let mut y_local = DenseMatrix::<Complex64>::zeros(src_dofs.len(), nrhs);
                block.matrix.matmul_into(&x_local, &mut y_local);
                for (i_local, &i_global) in src_dofs.iter().enumerate() {
                    for rhs in 0..nrhs {
                        // SAFETY: source clusters occur in exactly one
                        // forward partition, so each output row is owned by
                        // one worker. The row-major layout is part of the
                        // DenseMatrix contract.
                        unsafe {
                            *y_ptr.add(i_global * nrhs + rhs) += y_local[[i_local, rhs]];
                        }
                    }
                }
            }

            for block in &self.low_rank_blocks[partition.low_rank_start..partition.low_rank_end] {
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];
                let x_local =
                    DenseMatrix::from_fn(fld_dofs.len(), nrhs, |j, rhs| x[[fld_dofs[j], rhs]]);
                let mut y_local = DenseMatrix::<Complex64>::zeros(src_dofs.len(), nrhs);
                block.factors.apply_block_into(&x_local, &mut y_local);
                for (i_local, &i_global) in src_dofs.iter().enumerate() {
                    for rhs in 0..nrhs {
                        // SAFETY: see the dense-block scatter above.
                        unsafe {
                            *y_ptr.add(i_global * nrhs + rhs) += y_local[[i_local, rhs]];
                        }
                    }
                }
            }
        };
        if use_parallel(self.num_dofs.saturating_mul(nrhs), BEM_ENTRIES_PER_THREAD) {
            self.partitions.par_iter().for_each(scatter_partition);
        } else {
            self.partitions.iter().for_each(scatter_partition);
        }
    }

    /// Checked multi-right-hand-side forward H-matrix product.
    pub fn matvec_block_checked(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, HMatrixOperatorError> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_block_into_checked(x, &mut y)?;
        Ok(y)
    }

    /// Checked allocation-free multi-right-hand-side forward product.
    pub fn matvec_block_into_checked(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &mut DenseMatrix<Complex64>,
    ) -> Result<(), HMatrixOperatorError> {
        self.validate_block_input(x)?;
        self.validate_block_output_shape(x, y)?;
        self.matvec_block_into(x, y);
        self.validate_block_output(y)
    }

    /// Compute the algebraic transpose product y = A^T*x into a pre-allocated buffer.
    ///
    /// Partition-parallel over the field-cluster schedule
    /// ([`Self::transpose_partitions`]): each partition owns disjoint output
    /// (field) DOFs, so this mirrors the forward [`Self::matvec_into`]
    /// gather/compute/scatter pattern rather than serialising.
    pub fn matvec_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matvec_adjoint_like_into(x, y, AdjointKind::Transpose);
    }

    /// Compute the Hermitian (conjugate transpose) product y = A^H*x into a
    /// pre-allocated buffer. Same field-cluster schedule as
    /// [`Self::matvec_transpose_into`]; conjugation is folded into the block
    /// kernels (dense `ConjTrans` GEMV, conjugated low-rank factors).
    /// Compute algebraic-transpose H-matrix products for several right-hand
    /// sides in one field-cluster traversal.
    pub fn matvec_transpose_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_transpose_block_into(x, &mut y);
        y
    }

    /// Compute `Y = H^T · X` into a pre-allocated row-major dense matrix.
    pub fn matvec_transpose_block_into(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(x.nrows(), self.num_dofs);
        assert_eq!(y.nrows(), self.num_dofs);
        assert_eq!(y.ncols(), x.ncols());
        y.fill(Complex64::new(0.0, 0.0));

        let nrhs = x.ncols();
        let y_ptr = ScatterPtr(y.as_mut_slice().as_mut_ptr());
        // Transpose block scatter (rayon above the machine-adaptive gate,
        // serial for small systems). The partition closure is shared by both
        // paths. Work is output entries × right-hand sides.
        let scatter_partition = |partition: &HMatrixTransposePartition| {
            for &block_idx in
                &self.transpose_dense_order[partition.dense_start..partition.dense_end]
            {
                let block = &self.dense_blocks[block_idx];
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];
                let x_local =
                    DenseMatrix::from_fn(src_dofs.len(), nrhs, |i, rhs| x[[src_dofs[i], rhs]]);
                let mut y_local = DenseMatrix::<Complex64>::zeros(fld_dofs.len(), nrhs);
                let transposed = block.matrix.transposed();
                transposed.matmul_into(&x_local, &mut y_local);
                for (j_local, &j_global) in fld_dofs.iter().enumerate() {
                    for rhs in 0..nrhs {
                        unsafe {
                            *y_ptr.add(j_global * nrhs + rhs) += y_local[[j_local, rhs]];
                        }
                    }
                }
            }

            for &block_idx in
                &self.transpose_low_rank_order[partition.low_rank_start..partition.low_rank_end]
            {
                let block = &self.low_rank_blocks[block_idx];
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];
                let x_local =
                    DenseMatrix::from_fn(src_dofs.len(), nrhs, |i, rhs| x[[src_dofs[i], rhs]]);
                let mut y_local = DenseMatrix::<Complex64>::zeros(fld_dofs.len(), nrhs);
                block
                    .factors
                    .apply_transpose_block_into(&x_local, &mut y_local);
                for (j_local, &j_global) in fld_dofs.iter().enumerate() {
                    for rhs in 0..nrhs {
                        unsafe {
                            *y_ptr.add(j_global * nrhs + rhs) += y_local[[j_local, rhs]];
                        }
                    }
                }
            }
        };
        if use_parallel(self.num_dofs.saturating_mul(nrhs), BEM_ENTRIES_PER_THREAD) {
            self.transpose_partitions
                .par_iter()
                .for_each(scatter_partition);
        } else {
            self.transpose_partitions.iter().for_each(scatter_partition);
        }
    }

    /// Checked multi-right-hand-side algebraic-transpose product.
    pub fn matvec_transpose_block_checked(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, HMatrixOperatorError> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_transpose_block_into_checked(x, &mut y)?;
        Ok(y)
    }

    /// Checked allocation-free multi-right-hand-side transpose product.
    pub fn matvec_transpose_block_into_checked(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &mut DenseMatrix<Complex64>,
    ) -> Result<(), HMatrixOperatorError> {
        self.validate_block_input(x)?;
        self.validate_block_output_shape(x, y)?;
        self.matvec_transpose_block_into(x, y);
        self.validate_block_output(y)
    }

    /// Compute Hermitian (conjugate transpose) product y = A^H*x into a
    /// pre-allocated buffer. Same field-cluster schedule as
    /// [`Self::matvec_transpose_into`]; conjugation is folded into block
    /// kernels (dense `ConjTrans` GEMV, conjugated low-rank factors).
    pub fn matvec_hermitian_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matvec_adjoint_like_into(x, y, AdjointKind::Hermitian);
    }

    /// Allocating convenience for [`Self::matvec_hermitian_into`].
    pub fn matvec_hermitian(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_hermitian_into(x, &mut y);
        y
    }

    /// Compute Hermitian-transpose products for several right-hand sides.
    ///
    /// The scalar Hermitian traversal is reused per column until a fused
    /// conjugate-transpose block kernel is available. The checked wrapper
    /// below still provides the same finite-value and shape contract as the
    /// forward and algebraic-transpose block paths.
    pub fn matvec_hermitian_block_into(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(x.nrows(), self.num_dofs);
        assert_eq!(y.nrows(), self.num_dofs);
        assert_eq!(y.ncols(), x.ncols());
        y.fill(Complex64::new(0.0, 0.0));
        for rhs in 0..x.ncols() {
            let input: Vec<_> = (0..self.num_dofs).map(|row| x[[row, rhs]]).collect();
            let mut output = vec![Complex64::new(0.0, 0.0); self.num_dofs];
            self.matvec_hermitian_into(&input, &mut output);
            for (row, value) in output.into_iter().enumerate() {
                y[[row, rhs]] = value;
            }
        }
    }

    /// Checked multi-right-hand-side Hermitian-transpose product.
    pub fn matvec_hermitian_block_checked(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, HMatrixOperatorError> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_hermitian_block_into_checked(x, &mut y)?;
        Ok(y)
    }

    /// Checked allocation-free multi-right-hand-side Hermitian product.
    pub fn matvec_hermitian_block_into_checked(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &mut DenseMatrix<Complex64>,
    ) -> Result<(), HMatrixOperatorError> {
        self.validate_block_input(x)?;
        self.validate_block_output_shape(x, y)?;
        self.matvec_hermitian_block_into(x, y);
        self.validate_block_output(y)
    }

    /// Compare one H-matrix action against independently assembled outputs.
    ///
    /// `reference_outputs` must be generated without reusing this compressed
    /// operator (for example, from a dense TBEM matrix or a matrix-free exact
    /// kernel). The returned value is the maximum 2-norm relative error over
    /// all deterministic probes. Forward, algebraic-transpose, and Hermitian
    /// actions share the same checked dimension and finite-value boundary.
    pub fn relative_action_error(
        &self,
        action: HMatrixAction,
        probes: &[Vec<Complex64>],
        reference_outputs: &[Vec<Complex64>],
    ) -> Result<f64, HMatrixAccuracyError> {
        if probes.is_empty() {
            return Err(HMatrixAccuracyError::EmptyProbeSet);
        }
        if probes.len() != reference_outputs.len() {
            return Err(HMatrixAccuracyError::ProbeCountMismatch {
                probes: probes.len(),
                references: reference_outputs.len(),
            });
        }

        let mut maximum_error = 0.0_f64;
        for (probe_index, (probe, reference)) in probes.iter().zip(reference_outputs).enumerate() {
            let actual = match action {
                HMatrixAction::Forward => self.matvec_checked(probe),
                HMatrixAction::Transpose => self.matvec_transpose_checked(probe),
                HMatrixAction::Hermitian => self.matvec_hermitian_checked(probe),
            }
            .map_err(HMatrixAccuracyError::Operator)?;

            if reference.len() != self.num_dofs {
                return Err(HMatrixAccuracyError::ReferenceLength {
                    probe: probe_index,
                    expected: self.num_dofs,
                    actual: reference.len(),
                });
            }
            for (index, value) in reference.iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(HMatrixAccuracyError::NonFiniteReference {
                        probe: probe_index,
                        index,
                    });
                }
            }

            let difference_squared = actual
                .iter()
                .zip(reference)
                .map(|(value, expected)| (*value - *expected).norm_sqr())
                .sum::<f64>();
            let reference_squared = reference.iter().map(|value| value.norm_sqr()).sum::<f64>();
            let relative_error =
                difference_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE);
            if !relative_error.is_finite() {
                return Err(HMatrixAccuracyError::NonFiniteError { probe: probe_index });
            }
            maximum_error = maximum_error.max(relative_error);
        }

        Ok(maximum_error)
    }

    /// Checked algebraic-transpose H-matrix product.
    pub fn matvec_transpose_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, HMatrixOperatorError> {
        self.validate_operator_input(x)?;
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_transpose_into(x, &mut y);
        self.validate_operator_output(&y)?;
        Ok(y)
    }

    /// Checked Hermitian-transpose H-matrix product.
    pub fn matvec_hermitian_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, HMatrixOperatorError> {
        self.validate_operator_input(x)?;
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_hermitian_into(x, &mut y);
        self.validate_operator_output(&y)?;
        Ok(y)
    }

    fn validate_operator_input(&self, x: &[Complex64]) -> Result<(), HMatrixOperatorError> {
        if x.len() != self.num_dofs {
            return Err(HMatrixOperatorError::InputLength {
                expected: self.num_dofs,
                actual: x.len(),
            });
        }
        if let Some(index) = x
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HMatrixOperatorError::NonFiniteValue { index });
        }
        Ok(())
    }

    fn validate_block_input(&self, x: &DenseMatrix<Complex64>) -> Result<(), HMatrixOperatorError> {
        if x.nrows() != self.num_dofs {
            return Err(HMatrixOperatorError::BlockInputRows {
                expected: self.num_dofs,
                actual: x.nrows(),
            });
        }
        if let Some(index) = x
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HMatrixOperatorError::NonFiniteValue { index });
        }
        Ok(())
    }

    fn validate_block_output_shape(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &DenseMatrix<Complex64>,
    ) -> Result<(), HMatrixOperatorError> {
        if y.nrows() != self.num_dofs {
            return Err(HMatrixOperatorError::BlockOutputRows {
                expected: self.num_dofs,
                actual: y.nrows(),
            });
        }
        if y.ncols() != x.ncols() {
            return Err(HMatrixOperatorError::BlockColumnCount {
                input: x.ncols(),
                output: y.ncols(),
            });
        }
        Ok(())
    }

    fn validate_block_output(
        &self,
        y: &DenseMatrix<Complex64>,
    ) -> Result<(), HMatrixOperatorError> {
        if let Some(index) = y
            .as_slice()
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HMatrixOperatorError::NonFiniteValue { index });
        }
        Ok(())
    }

    fn validate_operator_output(&self, y: &[Complex64]) -> Result<(), HMatrixOperatorError> {
        if let Some(index) = y
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HMatrixOperatorError::NonFiniteValue { index });
        }
        Ok(())
    }

    /// Shared driver for transpose / Hermitian matvecs over the
    /// field-cluster partition schedule.
    fn matvec_adjoint_like_into(&self, x: &[Complex64], y: &mut [Complex64], kind: AdjointKind) {
        assert_eq!(x.len(), self.num_dofs);
        assert_eq!(y.len(), self.num_dofs);
        y.fill(Complex64::new(0.0, 0.0));

        // Every field cluster is assigned to exactly one transpose partition,
        // and clusters own disjoint DOF sets.  Contributions from blocks in a
        // single partition are still accumulated serially by that worker;
        // partitions can therefore scatter directly into the caller's output
        // without a per-partition Vec or a serial merge pass.
        //
        // SAFETY: the partition construction groups each field cluster once
        // (see `transpose_partitions`), and `cluster_dof_indices` is the
        // one-to-one cluster-to-DOF mapping used by the H-matrix assembly.
        // Thus no two Rayon workers write the same `y[j_global]`. `ScatterPtr`
        // makes that invariant explicit to Rayon; no aliasing mutable slice is
        // created and each location is updated by only its owning partition.
        let y_ptr = ScatterPtr(y.as_mut_ptr());
        // Adjoint scatter (rayon above the machine-adaptive gate, serial for
        // small systems). The partition closure is shared by both paths.
        // Work is output entries.
        let scatter_partition = |partition: &HMatrixTransposePartition| {
            HMATVEC_SCRATCH.with(|cell| {
                let scratch = &mut *cell.borrow_mut();

                // Dense blocks: input lives on the source (row) DOFs,
                // output on the field (column) DOFs.
                for &block_idx in
                    &self.transpose_dense_order[partition.dense_start..partition.dense_end]
                {
                    let block = &self.dense_blocks[block_idx];
                    let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                    let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

                    scratch.gather(x, src_dofs);
                    scratch.zero_y(fld_dofs.len());
                    match kind {
                        AdjointKind::Transpose => block
                            .matrix
                            .matvec_transpose_into(&scratch.x_local, &mut scratch.y_local),
                        AdjointKind::Hermitian => block
                            .matrix
                            .matvec_hermitian_into(&scratch.x_local, &mut scratch.y_local),
                    }

                    for (j_local, &j_global) in fld_dofs.iter().enumerate() {
                        // SAFETY: see the partition invariant above.
                        unsafe { *y_ptr.add(j_global) += scratch.y_local[j_local] };
                    }
                }

                for &block_idx in
                    &self.transpose_low_rank_order[partition.low_rank_start..partition.low_rank_end]
                {
                    let block = &self.low_rank_blocks[block_idx];
                    let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                    let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

                    scratch.gather(x, src_dofs);
                    scratch.zero_y(fld_dofs.len());
                    match kind {
                        AdjointKind::Transpose => block.factors.apply_transpose_into(
                            &scratch.x_local,
                            &mut scratch.y_local,
                            &mut scratch.t,
                        ),
                        AdjointKind::Hermitian => block.factors.apply_hermitian_into(
                            &scratch.x_local,
                            &mut scratch.y_local,
                            &mut scratch.t,
                        ),
                    }

                    for (j_local, &j_global) in fld_dofs.iter().enumerate() {
                        // SAFETY: see the partition invariant above.
                        unsafe { *y_ptr.add(j_global) += scratch.y_local[j_local] };
                    }
                }
            });
        };
        if use_parallel(self.num_dofs, BEM_ENTRIES_PER_THREAD) {
            self.transpose_partitions
                .par_iter()
                .for_each(scatter_partition);
        } else {
            self.transpose_partitions.iter().for_each(scatter_partition);
        }
    }
}

/// Block-kernel selector for [`HMatrix::matvec_adjoint_like_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdjointKind {
    /// Algebraic transpose `A^T` (no conjugation).
    Transpose,
    /// Hermitian conjugate transpose `A^H`.
    Hermitian,
}

impl LinearOperator<Complex64> for HMatrix {
    fn num_rows(&self) -> usize {
        self.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.num_dofs
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_into(x, &mut y);
        y
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matvec_into(x, y);
    }

    fn apply_block(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        self.matvec_block_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_transpose_into(x, &mut y);
        y
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matvec_transpose_into(x, y);
    }

    /// Hermitian apply routed through the parallel field-cluster schedule —
    /// avoids the trait default's conjugate round-trip through the serial
    /// transpose plus its two `O(n)` allocation/conjugation passes.
    fn apply_hermitian(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matvec_hermitian(x)
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::assembly::tbem::{
        HMatrixVelocityOperator, TbemVelocityOperatorError, build_tbem_rhs_stabilized_with_beta,
    };
    use crate::core::types::{Element, ElementProperty, ElementType};

    #[test]
    fn curved_quad4_dof_proxies_carry_collocation_normals() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.4],
            [0.0, 1.0, 0.0],
        ];
        let element = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 4],
            center: [0.5, 0.5, 0.1],
            area: 1.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2, 3],
        };
        let expected = element.collocation_normals(BasisOrder::Linear, &nodes);
        let (proxies, _) = build_dof_proxies(std::slice::from_ref(&element), &nodes);
        assert_eq!(proxies.len(), 4);
        assert!((expected[0][2] - expected[1][2]).abs() > 1.0e-6);
        for (proxy, normal) in proxies.iter().zip(expected) {
            let error = (0..3)
                .map(|axis| (proxy.normal[axis] - normal[axis]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error < 1.0e-12,
                "curved H-matrix proxy normal error={error}"
            );
        }
    }

    #[test]
    fn public_builder_matches_explicit_and_dense_across_ka_transition() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let speed_of_sound = 343.0;
        for ka in [0.49, 0.5, 0.51] {
            let frequency = ka * speed_of_sound / std::f64::consts::TAU;
            let physics = PhysicsParams::new(frequency, speed_of_sound, 1.21, false);
            let sign = PhysicsParams::optimal_formulation_sign(ka);
            let public = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-10, 16);
            let explicit = build_hmatrix_stabilized(
                &elements, &nodes, &clusters, &physics, 1e-10, 16, sign, 1.0,
            );
            let dense = crate::core::assembly::tbem::build_tbem_system_stabilized(
                &elements, &nodes, &physics, sign, 1.0,
            );

            let x: Vec<Complex64> = (0..elements.len())
                .map(|i| Complex64::new((0.37 * (i + 1) as f64).sin(), (0.19 * i as f64).cos()))
                .collect();
            let y_dense = dense.matrix.matvec(&x);
            let y_public = public.matvec(&x);
            let y_explicit = explicit.matvec(&x);

            let relative_error = |actual: &[Complex64], expected: &[Complex64]| {
                let diff = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let reference = expected
                    .iter()
                    .map(|value| value.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-30);
                diff / reference
            };

            let public_dense_error = relative_error(&y_public, &y_dense);
            let explicit_dense_error = relative_error(&y_explicit, &y_dense);
            let public_explicit_error = relative_error(&y_public, &y_explicit);
            assert!(
                public_dense_error < 1e-6,
                "public H-matrix builder diverged from dense TBEM at ka={ka}: rel={public_dense_error:.3e}"
            );
            assert!(
                explicit_dense_error < 1e-6,
                "explicit-sign H-matrix builder diverged from dense TBEM at ka={ka}: rel={explicit_dense_error:.3e}"
            );
            assert!(
                public_explicit_error < 1e-12,
                "public and explicit-sign H-matrix builders differ at ka={ka}: rel={public_explicit_error:.3e}"
            );
        }
    }

    /// Create a minimal 4-element mesh for testing H-matrix assembly
    fn make_test_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ];

        let make_elem = |conn: Vec<usize>, center: [f64; 3], dof: usize| Element {
            connectivity: conn,
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center,
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![dof],
        };

        let elements = vec![
            make_elem(vec![0, 1, 2], [0.5, 0.33, 0.0], 0),
            make_elem(vec![1, 3, 2], [1.17, 0.33, 0.0], 1),
            make_elem(vec![3, 4, 2], [1.83, 0.33, 0.0], 2),
            make_elem(vec![3, 5, 4], [2.17, 0.33, 0.33], 3),
        ];

        (elements, nodes)
    }

    /// Create minimal clusters: two clusters, each with 2 elements
    fn make_test_clusters() -> Vec<Cluster> {
        let mut c0 = Cluster::new([0.83, 0.33, 0.0]);
        c0.element_indices = vec![0, 1];
        c0.num_elements = 2;
        c0.radius = 1.0;
        c0.near_clusters = vec![0]; // self is near
        c0.far_clusters = vec![1]; // other cluster is far

        let mut c1 = Cluster::new([2.0, 0.33, 0.17]);
        c1.element_indices = vec![2, 3];
        c1.num_elements = 2;
        c1.radius = 1.0;
        c1.near_clusters = vec![1]; // self is near
        c1.far_clusters = vec![0]; // other cluster is far

        vec![c0, c1]
    }

    /// Two well-separated four-element patches. The larger far-field blocks
    /// make the derivative compression test distinguish a genuine low-rank
    /// factorisation from a full-rank 2×2 fixture.
    fn make_compression_test_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        for patch in 0..2 {
            let x_offset = patch as f64 * 12.0;
            for local in 0..8 {
                let x = x_offset + local as f64 * 1.1;
                let base = nodes.len();
                nodes.extend_from_slice(&[[x, 0.0, 0.0], [x + 0.8, 0.0, 0.0], [x, 0.8, 0.0]]);
                elements.push(Element {
                    connectivity: vec![base, base + 1, base + 2],
                    element_type: ElementType::Tri3,
                    property: ElementProperty::Surface,
                    normal: [0.0, 0.0, 1.0],
                    node_normals: vec![[0.0; 3]; 3],
                    center: [x + 0.8 / 3.0, 0.8 / 3.0, 0.0],
                    area: 0.32,
                    boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
                    group: 0,
                    dof_addresses: vec![patch * 8 + local],
                });
            }
        }
        (elements, nodes)
    }

    fn make_compression_test_clusters() -> Vec<Cluster> {
        let mut left = Cluster::new([4.12, 0.27, 0.0]);
        left.element_indices = (0..8).collect();
        left.num_elements = 8;
        left.radius = 4.5;
        left.near_clusters = vec![0];
        left.far_clusters = vec![1];

        let mut right = Cluster::new([16.12, 0.27, 0.0]);
        right.element_indices = (8..16).collect();
        right.num_elements = 8;
        right.radius = 4.5;
        right.near_clusters = vec![1];
        right.far_clusters = vec![0];

        vec![left, right]
    }

    #[test]
    fn higher_order_hmatrix_matches_linear_tbem_on_far_blocks() {
        let (mut elements, nodes) = make_test_mesh();
        for (element_index, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = (element_index * 3..element_index * 3 + 3).collect();
        }
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let hmatrix = build_hmatrix_stabilized(
            &elements, &nodes, &clusters, &physics, 1.0e-10, 16, sign, 0.2,
        );
        assert_eq!(hmatrix.num_dofs, 12);
        let has_dof_sized_far_block = hmatrix.low_rank_blocks.iter().any(|block| {
            block.source_cluster != block.field_cluster
                && block.factors.n_src() > 2
                && block.factors.n_fld() > 2
        }) || hmatrix.dense_blocks.iter().any(|block| {
            block.source_cluster != block.field_cluster
                && block.matrix.nrows() > 2
                && block.matrix.ncols() > 2
        });
        assert!(
            has_dof_sized_far_block,
            "higher-order far block must be stored with DOF dimensions"
        );

        let dense = crate::core::assembly::tbem::build_tbem_system_linear_stabilized(
            &elements, &nodes, &physics, sign, 0.2,
        )
        .matrix;
        let x: Vec<Complex64> = (0..12)
            .map(|index| Complex64::new(0.2 + index as f64 * 0.07, -0.1 + index as f64 * 0.03))
            .collect();
        let y_dense = dense.matvec(&x);
        let y_hmatrix = hmatrix.matvec(&x);
        let error = y_dense
            .iter()
            .zip(&y_hmatrix)
            .map(|(expected, actual)| (*expected - *actual).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = y_dense
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
        assert!(
            error / scale < 1.0e-8,
            "higher-order H-matrix parity error={}",
            error / scale
        );
    }

    #[test]
    fn higher_order_hmatrix_matches_quadratic_tbem_on_far_blocks() {
        let (mut elements, nodes) = make_test_mesh();
        for (element_index, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = (element_index * 6..element_index * 6 + 6).collect();
        }
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let hmatrix = build_hmatrix_stabilized(
            &elements, &nodes, &clusters, &physics, 1.0e-9, 16, sign, 0.2,
        );
        assert_eq!(hmatrix.num_dofs, 24);
        let has_dof_sized_far_block = hmatrix.low_rank_blocks.iter().any(|block| {
            block.source_cluster != block.field_cluster
                && block.factors.n_src() > 2
                && block.factors.n_fld() > 2
        }) || hmatrix.dense_blocks.iter().any(|block| {
            block.source_cluster != block.field_cluster
                && block.matrix.nrows() > 2
                && block.matrix.ncols() > 2
        });
        assert!(
            has_dof_sized_far_block,
            "higher-order far block must be stored with DOF dimensions"
        );

        let dense = crate::core::assembly::tbem::build_tbem_system_quadratic_stabilized(
            &elements, &nodes, &physics, sign, 0.2,
        )
        .matrix;
        let x: Vec<Complex64> = (0..24)
            .map(|index| Complex64::new(0.2 + index as f64 * 0.03, -0.1 + index as f64 * 0.02))
            .collect();
        let y_dense = dense.matvec(&x);
        let y_hmatrix = hmatrix.matvec(&x);
        let error = y_dense
            .iter()
            .zip(&y_hmatrix)
            .map(|(expected, actual)| (*expected - *actual).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = y_dense
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
        assert!(
            error / scale < 1.0e-7,
            "quadratic H-matrix parity error={}",
            error / scale
        );
    }

    fn make_quad4_hmatrix_mesh() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>) {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ];
        let element = |connectivity: Vec<usize>, dof_addresses: Vec<usize>, z: f64| Element {
            connectivity,
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 4],
            center: [0.5, 0.5, z],
            area: 1.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses,
        };
        let elements = vec![
            element(vec![0, 1, 2, 3], vec![0, 1, 2, 3], 0.0),
            element(vec![4, 5, 6, 7], vec![4, 5, 6, 7], 3.0),
        ];
        let mut first = Cluster::new([0.5, 0.5, 0.0]);
        first.element_indices = vec![0];
        first.num_elements = 1;
        first.radius = 1.0;
        first.near_clusters = vec![0];
        first.far_clusters = vec![1];
        let mut second = Cluster::new([0.5, 0.5, 3.0]);
        second.element_indices = vec![1];
        second.num_elements = 1;
        second.radius = 1.0;
        second.near_clusters = vec![1];
        second.far_clusters = vec![0];
        (elements, nodes, vec![first, second])
    }

    #[test]
    fn quad4_linear_hmatrix_matches_tbem() {
        let (elements, nodes, clusters) = make_quad4_hmatrix_mesh();
        let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let hmatrix = build_hmatrix_stabilized(
            &elements, &nodes, &clusters, &physics, 1.0e-10, 16, sign, 0.2,
        );
        assert_eq!(hmatrix.num_dofs, 8);
        assert!(
            hmatrix.low_rank_blocks.iter().any(|block| {
                block.source_cluster != block.field_cluster
                    && block.factors.n_src() > 2
                    && block.factors.n_fld() > 2
            }) || hmatrix.dense_blocks.iter().any(|block| {
                block.source_cluster != block.field_cluster
                    && block.matrix.nrows() > 2
                    && block.matrix.ncols() > 2
            }),
            "Quad4 linear far block must retain DOF dimensions"
        );
        let dense = crate::core::assembly::tbem::build_tbem_system_linear_stabilized(
            &elements, &nodes, &physics, sign, 0.2,
        )
        .matrix;
        let x: Vec<Complex64> = (0..8)
            .map(|index| Complex64::new(0.1 + index as f64 * 0.03, -0.2 + index as f64 * 0.02))
            .collect();
        let y_dense = dense.matvec(&x);
        let y_hmatrix = hmatrix.matvec(&x);
        let error = y_dense
            .iter()
            .zip(&y_hmatrix)
            .map(|(expected, actual)| (*expected - *actual).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = y_dense
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
        assert!(
            error / scale < 1.0e-8,
            "Quad4 linear H-matrix parity error={error}"
        );
    }

    #[test]
    fn quad4_quadratic_hmatrix_matches_tbem() {
        let (mut elements, nodes, clusters) = make_quad4_hmatrix_mesh();
        elements[0].dof_addresses = (0..9).collect();
        elements[1].dof_addresses = (9..18).collect();
        let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let hmatrix = build_hmatrix_stabilized(
            &elements, &nodes, &clusters, &physics, 1.0e-9, 16, sign, 0.2,
        );
        assert_eq!(hmatrix.num_dofs, 18);
        assert!(
            hmatrix.low_rank_blocks.iter().any(|block| {
                block.source_cluster != block.field_cluster
                    && block.factors.n_src() > 2
                    && block.factors.n_fld() > 2
            }) || hmatrix.dense_blocks.iter().any(|block| {
                block.source_cluster != block.field_cluster
                    && block.matrix.nrows() > 2
                    && block.matrix.ncols() > 2
            }),
            "Quad4 quadratic far block must retain DOF dimensions"
        );
        let dense = crate::core::assembly::tbem::build_tbem_system_quadratic_stabilized(
            &elements, &nodes, &physics, sign, 0.2,
        )
        .matrix;
        let x: Vec<Complex64> = (0..18)
            .map(|index| Complex64::new(0.1 + index as f64 * 0.02, -0.2 + index as f64 * 0.015))
            .collect();
        let y_dense = dense.matvec(&x);
        let y_hmatrix = hmatrix.matvec(&x);
        let error = y_dense
            .iter()
            .zip(&y_hmatrix)
            .map(|(expected, actual)| (*expected - *actual).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = y_dense
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
        assert!(
            error / scale < 1.0e-7,
            "Quad4 quadratic H-matrix parity error={error}"
        );
    }

    #[test]
    fn test_compute_matrix_entry() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        // Compute off-diagonal entry between element 0 and element 3 (well-separated)
        let entry = compute_matrix_entry(
            0,
            3,
            &elements[0],
            &elements[3],
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
        );

        // Entry should be non-zero for well-separated elements
        assert!(
            entry.norm() > 1e-20,
            "Matrix entry should be non-zero, got {entry}"
        );
    }

    #[test]
    fn regular_matrix_entry_wavenumber_derivative_is_typed_and_finite() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let derivative = compute_regular_matrix_entry_wavenumber_derivative(
            &elements[0],
            &elements[3],
            &nodes,
            &physics,
            gamma,
            tau,
            physics.burton_miller_beta(),
            physics.burton_miller_beta_wavenumber_derivative(),
            -1.0,
            6,
        )
        .expect("positive quadrature order must be accepted");
        assert!(derivative.is_finite());
        assert!(matches!(
            compute_regular_matrix_entry_wavenumber_derivative(
                &elements[0],
                &elements[3],
                &nodes,
                &physics,
                gamma,
                tau,
                physics.burton_miller_beta(),
                physics.burton_miller_beta_wavenumber_derivative(),
                -1.0,
                0,
            ),
            Err(MatrixDerivativeError::InvalidQuadratureOrder)
        ));
    }

    #[test]
    fn full_matrix_entry_wavenumber_derivative_covers_singular_self_term() {
        let (elements, nodes) = make_test_mesh();
        let speed_of_sound = 343.0;
        let density = 1.21;
        let base = PhysicsParams::new(100.0, speed_of_sound, density, false);
        let step = 1e-4;
        let plus_physics = PhysicsParams::new(
            (base.wave_number + step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let minus_physics = PhysicsParams::new(
            (base.wave_number - step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let gamma = Complex64::new(base.gamma(), 0.0);
        let tau = Complex64::new(base.tau, 0.0);
        let analytic = compute_matrix_entry_wavenumber_derivative(
            0,
            0,
            &elements[0],
            &elements[0],
            &nodes,
            &base,
            gamma,
            tau,
            base.burton_miller_beta(),
            base.burton_miller_beta_wavenumber_derivative(),
            -1.0,
            6,
        )
        .expect("positive quadrature order must be accepted");
        let plus = compute_matrix_entry(
            0,
            0,
            &elements[0],
            &elements[0],
            &nodes,
            &plus_physics,
            Complex64::new(plus_physics.gamma(), 0.0),
            Complex64::new(plus_physics.tau, 0.0),
            plus_physics.burton_miller_beta(),
            -1.0,
        );
        let minus = compute_matrix_entry(
            0,
            0,
            &elements[0],
            &elements[0],
            &nodes,
            &minus_physics,
            Complex64::new(minus_physics.gamma(), 0.0),
            Complex64::new(minus_physics.tau, 0.0),
            minus_physics.burton_miller_beta(),
            -1.0,
        );
        let finite_difference = (plus - minus) / (2.0 * step);
        let scale = 1.0 + analytic.norm().max(finite_difference.norm());
        assert!(
            (analytic - finite_difference).norm() < 3e-4 * scale,
            "analytic={analytic} finite_difference={finite_difference}"
        );
    }

    #[test]
    fn test_exact_stabilized_matvec_matches_dense_system() {
        let (mut elements, nodes) = make_test_mesh();
        elements[0].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.1, 0.0)],
            admittance: Complex64::new(0.02, -0.01),
        };
        let physics = PhysicsParams::new(1200.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta_scale = 4.0;
        let dense = crate::core::assembly::tbem::build_tbem_system_stabilized(
            &elements, &nodes, &physics, sign, beta_scale,
        );
        let x = vec![
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.4, 0.3),
            Complex64::new(0.7, -0.1),
            Complex64::new(0.2, 0.8),
        ];
        let expected = dense.matrix.matvec(&x);
        let actual =
            apply_tbem_stabilized_exact(&elements, &nodes, &physics, sign, beta_scale, &x).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (*actual - *expected).norm() < 1e-11,
                "matvec mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn exact_wavenumber_derivative_matvec_matches_central_difference() {
        let (elements, nodes) = make_test_mesh();
        let speed_of_sound = 343.0;
        let density = 1.21;
        let base = PhysicsParams::new(120.0, speed_of_sound, density, false);
        let step = 1e-4;
        let plus_physics = PhysicsParams::new(
            (base.wave_number + step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let minus_physics = PhysicsParams::new(
            (base.wave_number - step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let x = vec![
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.4, 0.3),
            Complex64::new(0.7, -0.1),
            Complex64::new(0.2, 0.8),
        ];
        let analytic = apply_tbem_stabilized_exact_wavenumber_derivative(
            &elements, &nodes, &base, 1.0, 1.0, &x,
        )
        .expect("valid exact derivative matvec");
        let plus = apply_tbem_stabilized_exact(&elements, &nodes, &plus_physics, 1.0, 1.0, &x)
            .expect("valid plus matvec");
        let minus = apply_tbem_stabilized_exact(&elements, &nodes, &minus_physics, 1.0, 1.0, &x)
            .expect("valid minus matvec");
        for (index, ((analytic, plus), minus)) in analytic.iter().zip(&plus).zip(&minus).enumerate()
        {
            let finite_difference = (*plus - *minus) / (2.0 * step);
            let scale = 1.0 + analytic.norm().max(finite_difference.norm());
            assert!(
                (*analytic - finite_difference).norm() < 3e-4 * scale,
                "index={index} analytic={analytic} finite_difference={finite_difference}"
            );
        }
    }

    #[test]
    fn higher_order_hmatrix_wavenumber_derivative_matches_centered_difference() {
        let (mut elements, nodes) = make_compression_test_mesh();
        for (element_index, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = (element_index * 6..element_index * 6 + 6).collect();
        }
        let clusters = make_compression_test_clusters();
        let speed_of_sound = 343.0;
        let density = 1.21;
        let base = PhysicsParams::new(100.0, speed_of_sound, density, false);
        let step = 1.0e-3;
        let plus = PhysicsParams::new(
            (base.wave_number + step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let minus = PhysicsParams::new(
            (base.wave_number - step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let sign = 1.0;
        let beta_scale = 1.0;
        let derivative = build_hmatrix_stabilized_wavenumber_derivative(
            &elements,
            &nodes,
            &clusters,
            &base,
            1.0e-4,
            16,
            AcaVariant::Hca,
            LowRankPrecision::F64,
            sign,
            beta_scale,
            6,
        )
        .expect("higher-order analytic derivative assembly");
        let value_plus = build_hmatrix_stabilized(
            &elements, &nodes, &clusters, &plus, 1.0e-4, 16, sign, beta_scale,
        );
        let value_minus = build_hmatrix_stabilized(
            &elements, &nodes, &clusters, &minus, 1.0e-4, 16, sign, beta_scale,
        );

        assert_eq!(derivative.num_dofs, 96);
        assert!(
            derivative.low_rank_blocks.iter().any(|block| {
                let source_dofs = derivative.cluster_dof_indices[block.source_cluster].len();
                let field_dofs = derivative.cluster_dof_indices[block.field_cluster].len();
                block.rank < source_dofs.min(field_dofs)
            }),
            "higher-order derivative must retain a genuinely compressed far block"
        );

        let x: Vec<Complex64> = (0..derivative.num_dofs)
            .map(|index| Complex64::new(0.2 + index as f64 * 0.017, -0.1 + index as f64 * 0.013))
            .collect();
        let plus_action = value_plus.matvec(&x);
        let minus_action = value_minus.matvec(&x);
        let analytic = derivative.matvec(&x);
        for (index, ((analytic, plus), minus)) in analytic
            .iter()
            .zip(&plus_action)
            .zip(&minus_action)
            .enumerate()
        {
            let finite_difference = (*plus - *minus) / (2.0 * step);
            let scale = 1.0 + analytic.norm().max(finite_difference.norm());
            assert!(
                (*analytic - finite_difference).norm() < 2.0e-3 * scale,
                "index={index} analytic={analytic} finite_difference={finite_difference}"
            );
        }
    }

    #[test]
    fn hmatrix_derivative_reference_preserves_block_schedule_and_exact_values() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
        let hmatrix =
            build_hmatrix_stabilized(&elements, &nodes, &clusters, &physics, 1e-10, 16, 1.0, 1.0);
        let derivative = hmatrix
            .build_wavenumber_derivative_reference(&elements, &nodes, &physics, 1.0, 1.0, 6)
            .expect("reference derivative assembly");
        assert_eq!(derivative.dense_blocks.len(), hmatrix.dense_blocks.len());
        assert_eq!(
            derivative.low_rank_blocks.len(),
            hmatrix.low_rank_blocks.len()
        );
        assert_eq!(derivative.partitions.len(), hmatrix.partitions.len());

        let x = vec![
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.4, 0.3),
            Complex64::new(0.7, -0.1),
            Complex64::new(0.2, 0.8),
        ];
        let expected = apply_tbem_stabilized_exact_wavenumber_derivative(
            &elements, &nodes, &physics, 1.0, 1.0, &x,
        )
        .expect("exact derivative reference");
        let actual = derivative.matvec(&x);
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (*actual - *expected).norm() < 1e-10,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn compressed_wavenumber_derivative_matches_reference_and_stays_compressed() {
        let (elements, nodes) = make_compression_test_mesh();
        let clusters = make_compression_test_clusters();
        let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
        let derivative = build_hmatrix_stabilized_wavenumber_derivative(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-4,
            16,
            AcaVariant::Hca,
            LowRankPrecision::F64,
            1.0,
            1.0,
            6,
        )
        .expect("compressed derivative assembly");
        let value =
            build_hmatrix_stabilized(&elements, &nodes, &clusters, &physics, 1e-4, 16, 1.0, 1.0);
        let reference = value
            .build_wavenumber_derivative_reference(&elements, &nodes, &physics, 1.0, 1.0, 6)
            .expect("reference derivative assembly");

        assert!(
            !derivative.low_rank_blocks.is_empty(),
            "derivative build must retain at least one far-field low-rank block"
        );
        assert!(
            derivative.low_rank_blocks.iter().any(|block| {
                let m = derivative.cluster_dof_indices[block.source_cluster].len();
                let n = derivative.cluster_dof_indices[block.field_cluster].len();
                block.rank < m.min(n)
            }),
            "derivative build must exercise genuine rank compression"
        );

        let x = probe_vector(elements.len());
        let actual = derivative.matvec(&x);
        let expected = reference.matvec(&x);
        let forward_error = relative_vec_error(&actual, &expected);
        assert!(
            forward_error < 1e-7,
            "compressed derivative forward error = {forward_error:.3e}"
        );

        let actual_t = derivative.apply_transpose(&x);
        let expected_t = reference.apply_transpose(&x);
        let transpose_error = relative_vec_error(&actual_t, &expected_t);
        assert!(
            transpose_error < 1e-7,
            "compressed derivative transpose error = {transpose_error:.3e}"
        );

        let actual_h = derivative.matvec_hermitian(&x);
        let expected_h = reference.matvec_hermitian(&x);
        let hermitian_error = relative_vec_error(&actual_h, &expected_h);
        assert!(
            hermitian_error < 1e-7,
            "compressed derivative Hermitian error = {hermitian_error:.3e}"
        );
    }

    #[test]
    fn compressed_wavenumber_derivative_has_central_difference_taylor_slope() {
        let (elements, nodes) = make_compression_test_mesh();
        let clusters = make_compression_test_clusters();
        let speed_of_sound = 343.0;
        let base = PhysicsParams::new(120.0, speed_of_sound, 1.21, false);
        let step = 1e-4;
        let plus = PhysicsParams::new(
            (base.wave_number + step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            1.21,
            false,
        );
        let minus = PhysicsParams::new(
            (base.wave_number - step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            1.21,
            false,
        );
        let derivative =
            build_hmatrix_wavenumber_derivative(&elements, &nodes, &clusters, &base, 1e-4, 16, 6)
                .expect("compressed derivative assembly");
        let plus_h = build_hmatrix(&elements, &nodes, &clusters, &plus, 1e-4, 16);
        let minus_h = build_hmatrix(&elements, &nodes, &clusters, &minus, 1e-4, 16);
        let x = probe_vector(elements.len());
        let analytic = derivative.matvec(&x);
        let plus_y = plus_h.matvec(&x);
        let minus_y = minus_h.matvec(&x);
        let finite_difference: Vec<Complex64> = plus_y
            .iter()
            .zip(&minus_y)
            .map(|(p, m)| (*p - *m) / (2.0 * step))
            .collect();
        let error = relative_vec_error(&analytic, &finite_difference);
        assert!(
            error < 2e-3,
            "compressed derivative central-difference error = {error:.3e}"
        );
    }

    #[test]
    fn compressed_density_derivative_matches_central_difference() {
        let (mut elements, nodes) = make_compression_test_mesh();
        let clusters = make_compression_test_clusters();
        for element in &mut elements {
            element.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: Complex64::new(0.15, -0.02),
            };
        }
        let base = PhysicsParams::new(120.0, 343.0, 1.21, false);
        let step = base.density * 1.0e-5;
        let plus = PhysicsParams::new(120.0, 343.0, base.density + step, false);
        let minus = PhysicsParams::new(120.0, 343.0, base.density - step, false);
        let sign = PhysicsParams::optimal_formulation_sign(base.wave_number);
        let derivative = build_hmatrix_stabilized_density_derivative(
            &elements,
            &nodes,
            &clusters,
            &base,
            1.0e-4,
            16,
            AcaVariant::default(),
            LowRankPrecision::default(),
            sign,
            1.0,
        )
        .expect("compressed density derivative assembly");
        let plus_h =
            build_hmatrix_stabilized(&elements, &nodes, &clusters, &plus, 1.0e-4, 16, sign, 1.0);
        let minus_h =
            build_hmatrix_stabilized(&elements, &nodes, &clusters, &minus, 1.0e-4, 16, sign, 1.0);
        let x = probe_vector(elements.len());
        let analytic = derivative.matvec(&x);
        let plus_y = plus_h.matvec(&x);
        let minus_y = minus_h.matvec(&x);
        let finite_difference: Vec<_> = plus_y
            .iter()
            .zip(&minus_y)
            .map(|(p, m)| (*p - *m) / (2.0 * step))
            .collect();
        let error = relative_vec_error(&analytic, &finite_difference);
        assert!(
            error < 2.0e-3,
            "compressed density derivative central-difference error = {error:.3e}"
        );
    }

    #[test]
    fn compressed_quadratic_density_derivative_matches_central_difference() {
        let (mut elements, nodes, clusters) = make_quad4_hmatrix_mesh();
        elements[0].dof_addresses = (0..9).collect();
        elements[1].dof_addresses = (9..18).collect();
        for element in &mut elements {
            element.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.0, 0.0)],
                admittance: Complex64::new(0.15, -0.02),
            };
        }
        let base = PhysicsParams::new(120.0, 343.0, 1.21, false);
        let step = base.density * 1.0e-5;
        let plus = PhysicsParams::new(120.0, 343.0, base.density + step, false);
        let minus = PhysicsParams::new(120.0, 343.0, base.density - step, false);
        let sign = PhysicsParams::optimal_formulation_sign(base.wave_number);
        let derivative = build_hmatrix_stabilized_density_derivative(
            &elements,
            &nodes,
            &clusters,
            &base,
            1.0e-10,
            64,
            AcaVariant::default(),
            LowRankPrecision::default(),
            sign,
            1.0,
        )
        .expect("compressed quadratic density derivative assembly");
        let plus_h =
            build_hmatrix_stabilized(&elements, &nodes, &clusters, &plus, 1.0e-10, 64, sign, 1.0);
        let minus_h =
            build_hmatrix_stabilized(&elements, &nodes, &clusters, &minus, 1.0e-10, 64, sign, 1.0);
        let x = probe_vector(18);
        let analytic = derivative.matvec(&x);
        let plus_y = plus_h.matvec(&x);
        let minus_y = minus_h.matvec(&x);
        let finite_difference: Vec<_> = plus_y
            .iter()
            .zip(&minus_y)
            .map(|(p, m)| (*p - *m) / (2.0 * step))
            .collect();
        let error = relative_vec_error(&analytic, &finite_difference);
        assert!(
            error < 2.0e-3,
            "compressed quadratic density derivative central-difference error = {error:.3e}"
        );
    }

    #[test]
    fn test_velocity_with_admittance_changes_hmatrix_entry() {
        let (mut elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        elements[3].boundary_condition =
            BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        let rigid = compute_matrix_entry(
            0,
            3,
            &elements[0],
            &elements[3],
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
        );

        elements[3].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.0, 0.0)],
            admittance: Complex64::new(0.15, -0.02),
        };
        let with_admittance = compute_matrix_entry(
            0,
            3,
            &elements[0],
            &elements[3],
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
        );

        assert!(
            (with_admittance - rigid).norm() > 1e-12,
            "VelocityWithAdmittance must modify H-matrix entries"
        );
    }

    #[test]
    fn test_transfer_admittance_changes_hmatrix_entry() {
        let (mut elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        elements[3].boundary_condition =
            BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        let rigid = compute_matrix_entry(
            0,
            3,
            &elements[0],
            &elements[3],
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
        );

        elements[3].boundary_condition = BoundaryCondition::TransferAdmittance {
            admittance: Complex64::new(0.1, 0.0),
        };
        let with_transfer = compute_matrix_entry(
            0,
            3,
            &elements[0],
            &elements[3],
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
        );

        assert!(
            (with_transfer - rigid).norm() > 1e-12,
            "TransferAdmittance must modify H-matrix entries"
        );
    }

    #[test]
    fn test_aca_partial_pivot() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        let source_indices = vec![0, 1];
        let field_indices = vec![2, 3];

        let (u, v, rank, converged) = aca_partial_pivot(
            &source_indices,
            &field_indices,
            &elements,
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
            1e-4,
            2,
        );

        // Rank should be > 0 and <= min(m, n)
        assert!(rank > 0, "ACA should produce non-zero rank");
        assert!(rank <= 2, "Rank should be at most min(m, n) = 2");
        assert_eq!(u.nrows(), 2, "U should have 2 rows (source DOFs)");
        assert_eq!(v.nrows(), 2, "V should have 2 rows (field DOFs)");
        assert_eq!(u.ncols(), rank, "U columns should match rank");
        assert_eq!(v.ncols(), rank, "V columns should match rank");
        // With max_rank=2 == min(m,n)=2, the cap is geometric, so the result
        // is always considered converged regardless of the tolerance criterion.
        assert!(converged, "Geometric saturation must report converged=true");
    }

    #[test]
    fn test_build_hmatrix() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        assert_eq!(hmat.num_dofs, 4);
        // Should have 2 dense blocks (self-interactions) and up to 2 low-rank blocks
        assert!(
            !hmat.dense_blocks.is_empty(),
            "Should have at least one dense block"
        );
        assert!(hmat.memory_bytes > 0, "Memory should be non-zero");
        // For very small test meshes the H-matrix may use more storage than dense
        // due to block overhead. Just verify the ratio is computed and positive.
        assert!(
            hmat.compression_ratio > 0.0,
            "Compression ratio should be positive, got {}",
            hmat.compression_ratio
        );
        assert!(
            !hmat.partitions.is_empty(),
            "H-matrix should expose NUMA row partitions"
        );
        let partition_dofs: usize = hmat.partitions.iter().map(|p| p.num_dofs).sum();
        assert_eq!(partition_dofs, hmat.num_dofs);
    }

    #[test]
    fn test_hmatrix_partitions_cover_blocks_once() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        let dense_count: usize = hmat
            .partitions
            .iter()
            .map(|p| p.dense_end - p.dense_start)
            .sum();
        let low_rank_count: usize = hmat
            .partitions
            .iter()
            .map(|p| p.low_rank_end - p.low_rank_start)
            .sum();
        assert_eq!(dense_count, hmat.dense_blocks.len());
        assert_eq!(low_rank_count, hmat.low_rank_blocks.len());

        let mut covered_clusters: Vec<usize> = hmat
            .partitions
            .iter()
            .flat_map(|p| p.cluster_indices.iter().copied())
            .collect();
        covered_clusters.sort_unstable();
        assert_eq!(covered_clusters, vec![0, 1]);
    }

    #[test]
    fn test_morton_ordered_cluster_indices_are_spatial() {
        let clusters = vec![
            Cluster::new([1.0, 1.0, 1.0]),
            Cluster::new([0.0, 0.0, 0.0]),
            Cluster::new([0.5, 0.5, 0.5]),
        ];
        assert_eq!(morton_ordered_cluster_indices(&clusters), vec![1, 2, 0]);
    }

    #[test]
    fn test_aca_plus_basic() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        let source_indices = vec![0, 1];
        let field_indices = vec![2, 3];

        let (u, v, rank, converged) = aca_plus(
            &source_indices,
            &field_indices,
            &elements,
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
            1e-4,
            2,
        );

        assert!(rank > 0, "ACA+ should produce non-zero rank");
        assert!(rank <= 2);
        assert_eq!(u.nrows(), 2);
        assert_eq!(v.nrows(), 2);
        assert_eq!(u.ncols(), rank);
        assert_eq!(v.ncols(), rank);
        assert!(converged, "geometric saturation must report converged=true");
    }

    #[test]
    fn test_aca_gp_basic() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        let source_indices = vec![0, 1];
        let field_indices = vec![2, 3];

        let (u, v, rank, converged) = aca_gp(
            &source_indices,
            &field_indices,
            &elements,
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
            1e-4,
            2,
        );

        assert!(rank > 0, "ACA-GP should produce non-zero rank");
        assert!(rank <= 2);
        assert_eq!(u.nrows(), 2);
        assert_eq!(v.nrows(), 2);
        assert_eq!(u.ncols(), rank);
        assert_eq!(v.ncols(), rank);
        assert!(converged, "geometric saturation must report converged=true");
    }

    #[test]
    fn test_hca_basic() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        let source_indices = vec![0, 1];
        let field_indices = vec![2, 3];

        let (u, v, rank, converged) = hca(
            &source_indices,
            &field_indices,
            &elements,
            &nodes,
            &physics,
            gamma,
            tau,
            beta,
            -1.0,
            1e-4,
            2,
        );

        assert!(rank > 0, "HCA should produce non-zero rank");
        assert!(rank <= 2);
        assert_eq!(u.nrows(), 2);
        assert_eq!(v.nrows(), 2);
        assert_eq!(u.ncols(), rank);
        assert_eq!(v.ncols(), rank);
        assert!(converged, "geometric saturation must report converged=true");
    }

    #[test]
    fn test_centroid_anchor_is_innermost() {
        let (elements, _) = make_test_mesh();
        // `centroid_anchor` returns the *local* index (position within the
        // input slice) of the element whose center sits nearest the centroid.
        // Element centers along x: 0=0.50, 1=1.17, 2=1.83, 3=2.17.
        // For [0, 1, 2] the centroid x≈1.17 → element 1 wins; local index 1.
        // For [1, 2, 3] the centroid x≈1.72 → element 2 wins; local index 1.
        assert_eq!(centroid_anchor(&[0, 1, 2], &elements), 1);
        assert_eq!(centroid_anchor(&[1, 2, 3], &elements), 1);
    }

    #[test]
    fn test_build_hmatrix_variant_dispatch() {
        // All three variants must produce the same dense/low-rank topology
        // on the same fixture (variants only differ inside ACA).
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let h_pp = build_hmatrix_with_variant(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-4,
            10,
            AcaVariant::PartialPivot,
        );
        let h_plus = build_hmatrix_with_variant(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-4,
            10,
            AcaVariant::Plus,
        );
        let h_gp = build_hmatrix_with_variant(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-4,
            10,
            AcaVariant::Gp,
        );
        let h_hca = build_hmatrix_with_variant(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-4,
            10,
            AcaVariant::Hca,
        );
        let h_default = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        assert_eq!(h_pp.num_dofs, h_plus.num_dofs);
        assert_eq!(h_pp.num_dofs, h_gp.num_dofs);
        assert_eq!(h_pp.num_dofs, h_hca.num_dofs);
        assert_eq!(h_default.num_dofs, h_hca.num_dofs);
        assert_eq!(h_pp.dense_blocks.len(), h_plus.dense_blocks.len());
        assert_eq!(h_pp.dense_blocks.len(), h_gp.dense_blocks.len());
        assert_eq!(h_default.dense_blocks.len(), h_hca.dense_blocks.len());
        assert_eq!(h_pp.low_rank_blocks.len(), h_plus.low_rank_blocks.len());
        assert_eq!(h_pp.low_rank_blocks.len(), h_gp.low_rank_blocks.len());
        assert_eq!(h_default.low_rank_blocks.len(), h_hca.low_rank_blocks.len());

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ];
        for hmat in [&h_pp, &h_plus, &h_gp, &h_hca, &h_default] {
            let y = hmat.matvec(x.as_slice());
            let yn: f64 = y.iter().map(|v| v.norm()).sum();
            assert!(yn > 1e-15);
        }
    }

    #[test]
    fn test_build_hmatrix_thread_count_determinism() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);

        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool");
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-thread pool");

        let h1 =
            one_thread.install(|| build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-8, 16));
        let h4 = four_threads
            .install(|| build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-8, 16));

        assert_eq!(h1.num_dofs, h4.num_dofs);
        assert_eq!(h1.dense_blocks.len(), h4.dense_blocks.len());
        assert_eq!(h1.low_rank_blocks.len(), h4.low_rank_blocks.len());
        assert_eq!(h1.memory_bytes, h4.memory_bytes);

        let x = vec![
            Complex64::new(0.25, -0.5),
            Complex64::new(-0.75, 0.125),
            Complex64::new(1.25, 0.5),
            Complex64::new(-0.25, 0.875),
        ];
        let y1 = h1.matvec(x.as_slice());
        let y4 = h4.matvec(x.as_slice());
        for (a, b) in y1.iter().zip(y4.iter()) {
            assert!(
                (a - b).norm() <= 1e-12,
                "thread-count-dependent H-matrix matvec: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn test_hmatrix_rank_stats_empty() {
        // H-matrix with no low-rank blocks must report a zeroed RankStats.
        let hmat = HMatrix {
            num_dofs: 0,
            dense_blocks: Vec::new(),
            low_rank_blocks: Vec::new(),
            cluster_dof_indices: Vec::new(),
            partitions: Vec::new(),
            transpose_dense_order: Vec::new(),
            transpose_low_rank_order: Vec::new(),
            transpose_partitions: Vec::new(),
            memory_bytes: 0,
            compression_ratio: 1.0,
        };
        let stats = hmat.rank_stats();
        assert_eq!(stats.block_count, 0);
        assert_eq!(stats.saturated_count, 0);
        assert_eq!(stats.saturation_fraction, 0.0);
        assert_eq!(stats.mean_rank, 0.0);
    }

    #[test]
    fn test_hmatrix_rank_stats_summary() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        // Generous rank cap so any saturation here is genuinely the algorithm
        // running out of useful pivots, not the cap biting.
        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 64);
        let stats = hmat.rank_stats();

        // The two-cluster fixture yields two cross blocks.
        assert_eq!(stats.block_count, hmat.low_rank_blocks.len());
        assert!(
            stats.block_count >= 1,
            "expected at least one low-rank block"
        );
        assert!(stats.min_rank <= stats.max_rank);
        assert!(stats.mean_rank > 0.0);
        assert!(stats.mean_requested_max_rank >= stats.mean_rank);
        assert_eq!(
            stats.saturated_count, 0,
            "with max_rank=64 on a tiny fixture nothing should saturate"
        );
        assert_eq!(stats.saturation_fraction, 0.0);
    }

    /// Conjugation audit: assemble the dense 4×4 BEM block from
    /// `compute_matrix_entry` directly, then compare against
    /// `HMatrix::matvec` on a complex-valued input. If ACA stored U V^T
    /// while matvec applies U V^H, the two will disagree on this fixture
    /// (the kernel is complex with non-aligned per-entry phase).
    #[test]
    fn test_hmatrix_matvec_dense_parity_complex() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        // Pick a frequency where the Helmholtz kernel is genuinely complex
        // (non-real, non-aligned phase across entries).
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);

        // Assemble the full dense 4×4 reference using compute_matrix_entry.
        let n = elements
            .iter()
            .map(|e| e.dof_addresses.len())
            .sum::<usize>();
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        for (i, src) in elements.iter().enumerate() {
            for (j, fld) in elements.iter().enumerate() {
                dense[[i, j]] =
                    compute_matrix_entry(i, j, src, fld, &nodes, &physics, gamma, tau, beta, sign);
            }
        }

        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-10, 16);

        // Confirm the H-matrix factually contains low-rank cross blocks,
        // otherwise this is just a dense parity test in disguise.
        assert!(
            !hmat.low_rank_blocks.is_empty(),
            "audit fixture must exercise at least one low-rank block"
        );

        // Probe with a deterministic complex input.
        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let xr = (0.37 * (i + 1) as f64).sin();
                let xi = (0.19 * (i + 1) as f64).cos();
                Complex64::new(xr, xi)
            })
            .collect();

        let y_dense = dense.matvec(&x);
        let y_hmat = hmat.matvec(x.as_slice());

        let diff_norm: f64 = y_dense
            .iter()
            .zip(y_hmat.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let ref_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let rel = diff_norm / ref_norm.max(1e-30);

        // Permissive bound — ACA at tol=1e-10 with rank cap 16 on a 4×4
        // fixture should reconstruct each block to round-off. If this
        // assertion fails, the U V^T (build) vs U V^H (matvec) mismatch is
        // real and must be fixed before any operator preconditioner can be
        // trusted on top of the H-matrix path.
        assert!(
            rel < 1e-6,
            "H-matrix matvec must match dense matvec on a complex input \
             (relative error {rel:.3e}); likely conjugation convention mismatch \
             between ACA build and matvec apply"
        );
    }

    /// Phase 1.6: f32 storage of low-rank `(U, V)` factors. Build the same
    /// H-matrix in F64 and F32 storage modes, verify (a) memory drops to
    /// roughly half on the low-rank tail, (b) matvec parity vs dense stays
    /// within the published f32 relative-error band, (c) F64 mode still
    /// reconstructs to round-off (the strict gate).
    #[test]
    fn test_hmatrix_lowrank_f32_storage_parity_and_memory() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);

        let n = elements
            .iter()
            .map(|e| e.dof_addresses.len())
            .sum::<usize>();
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        for (i, src) in elements.iter().enumerate() {
            for (j, fld) in elements.iter().enumerate() {
                dense[[i, j]] =
                    compute_matrix_entry(i, j, src, fld, &nodes, &physics, gamma, tau, beta, sign);
            }
        }

        let h_f64 = build_hmatrix_with_variant_and_precision(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-10,
            16,
            AcaVariant::default(),
            LowRankPrecision::F64,
        );
        let h_f32 = build_hmatrix_with_variant_and_precision(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-10,
            16,
            AcaVariant::default(),
            LowRankPrecision::F32,
        );

        assert!(
            !h_f64.low_rank_blocks.is_empty(),
            "fixture must exercise at least one low-rank block"
        );
        assert_eq!(
            h_f64.low_rank_blocks.len(),
            h_f32.low_rank_blocks.len(),
            "precision must not change the cluster-pair classification"
        );
        for (b64, b32) in h_f64
            .low_rank_blocks
            .iter()
            .zip(h_f32.low_rank_blocks.iter())
        {
            assert_eq!(b64.rank, b32.rank);
            assert_eq!(b64.factors.precision(), LowRankPrecision::F64);
            assert_eq!(b32.factors.precision(), LowRankPrecision::F32);
            // F32 stores 8 bytes per entry + 16 bytes for the
            // (scale_u, scale_v) Higham-scaling pair; F64 stores 16
            // per entry with no scales. The factor-storage half + the
            // 16 B bookkeeping is the exact accounting.
            let entries = (b64.factors.n_src() + b64.factors.n_fld()) * b64.rank;
            assert_eq!(b64.factors.bytes(), entries * 16);
            assert_eq!(b32.factors.bytes(), entries * 8 + 16);
        }

        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let xr = (0.37 * (i + 1) as f64).sin();
                let xi = (0.19 * (i + 1) as f64).cos();
                Complex64::new(xr, xi)
            })
            .collect();
        let y_dense = dense.matvec(&x);
        let y_f64 = h_f64.matvec(&x);
        let y_f32 = h_f32.matvec(&x);

        let rel = |y: &[Complex64]| -> f64 {
            let diff: f64 = y_dense
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let r: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            diff / r.max(1e-30)
        };
        let rel_f64 = rel(&y_f64);
        let rel_f32 = rel(&y_f32);

        // F64 keeps the strict round-off bound used by the dense parity gate.
        assert!(
            rel_f64 < 1e-6,
            "F64 storage must match dense to round-off (rel = {rel_f64:.3e})"
        );
        // F32 storage forces a ≈1e-7 floor per block; on this 4×4 fixture
        // with a single low-rank pair, the realised global bound is well
        // inside 1e-5.
        assert!(
            rel_f32 < 1e-5,
            "F32 storage must stay inside the documented 1e-5 band \
             (rel = {rel_f32:.3e})"
        );
        // Sanity: F32 must NOT match round-off — otherwise the cast did
        // nothing and the test is silently passing.
        assert!(
            rel_f32 > rel_f64,
            "F32 path should be measurably less precise than F64 \
             (f32 = {rel_f32:.3e}, f64 = {rel_f64:.3e})"
        );

        // Apply_transpose path also exercised by the F32 upcast.
        let yt_f64 = <HMatrix as LinearOperator<Complex64>>::apply_transpose(&h_f64, x.as_slice());
        let yt_f32 = <HMatrix as LinearOperator<Complex64>>::apply_transpose(&h_f32, x.as_slice());
        let yt_diff: f64 = yt_f64
            .iter()
            .zip(yt_f32.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let yt_ref: f64 = yt_f64.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let yt_rel = yt_diff / yt_ref.max(1e-30);
        assert!(
            yt_rel < 1e-5,
            "F32 apply_transpose must stay inside 1e-5 of F64 (rel = {yt_rel:.3e})"
        );
    }

    /// Phase 1.6: per-block Higham-style scaling for `LowRankPrecision::F32`.
    ///
    /// Two precision regimes where scaling demonstrably beats the plain
    /// downcast — both are about getting blocks **into the f32 normal
    /// range** at storage time, not about saving precision during the
    /// matvec inner loop (the f64 accumulator already covers that).
    ///
    /// 1. **Subnormal-magnitude block** (`|·| ≈ 10⁻⁴⁰`): plain f32
    ///    downcast produces subnormal entries with reduced relative
    ///    precision (≈ 10⁻⁵ instead of f32's normal 10⁻⁷). Scaled path
    ///    stores entries near `1.0` (full f32 precision) and folds the
    ///    `10⁻⁴⁰` magnitude into the `f64` `scale_*` field. Reconstruction
    ///    matches the f64 reference to ~10⁻⁷.
    ///
    /// 2. **Overflow-magnitude block** (`|·| ≈ 10⁵⁰`, well over f32's
    ///    `≈ 3.4×10³⁸` ceiling): plain f32 downcast turns every entry
    ///    into `±Inf`, breaking the matvec entirely. Scaled path is
    ///    fully finite.
    ///
    /// The third assertion exercises the all-zero block edge case (scale
    /// must floor at `MIN_SCALE_F32` to keep the inverse finite).
    #[test]
    fn test_hmatrix_lowrank_f32_scaling_dynamic_range() {
        let m = 6;
        let n = 5;
        let rank = 1;

        // Helper: build a rank-1 block with given magnitude (constant
        // phase 0.6+0.8i so |·| = magnitude exactly), evaluate at a
        // deterministic complex input, return the per-row relative error
        // of (scaled, naive) F32 reconstructions vs the f64 reference.
        fn measure(
            m: usize,
            n: usize,
            magnitude: f64,
        ) -> (f64, f64, Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
            let mut u = DenseMatrix::<Complex64>::zeros(m, 1);
            for i in 0..m {
                let phase = 0.13_f64 * (i + 1) as f64;
                u[[i, 0]] = Complex64::new(magnitude * phase.cos(), magnitude * phase.sin());
            }
            let mut v = DenseMatrix::<Complex64>::zeros(n, 1);
            for j in 0..n {
                let phase = 0.37_f64 * (j + 1) as f64;
                v[[j, 0]] = Complex64::new(phase.cos(), phase.sin());
            }
            let x: Vec<Complex64> = (0..n)
                .map(|j| {
                    let xr = (0.41 * (j + 1) as f64).cos();
                    let xi = (0.23 * (j + 1) as f64).sin();
                    Complex64::new(xr, xi)
                })
                .collect();
            let f64_factors = LowRankFactors::F64 {
                u: u.clone(),
                v: v.clone(),
            };
            let scaled = downcast_factors_f64_to_f32(u.clone(), v.clone());
            let u32_data: Vec<Complex32> = u
                .as_slice()
                .iter()
                .map(|c| Complex32::new(c.re as f32, c.im as f32))
                .collect();
            let v32_data: Vec<Complex32> = v
                .as_slice()
                .iter()
                .map(|c| Complex32::new(c.re as f32, c.im as f32))
                .collect();
            let naive = LowRankFactors::F32 {
                u: DenseMatrix::from_row_major_vec(m, 1, u32_data),
                v: DenseMatrix::from_row_major_vec(n, 1, v32_data),
                scale_u: 1.0,
                scale_v: 1.0,
            };
            let y_f64 = f64_factors.apply(&x);
            let y_scaled = scaled.apply(&x);
            let y_naive = naive.apply(&x);
            let mut max_rel_scaled = 0.0_f64;
            let mut max_rel_naive = 0.0_f64;
            for i in 0..m {
                let denom = y_f64[i].norm().max(f64::MIN_POSITIVE);
                let rs = (y_f64[i] - y_scaled[i]).norm() / denom;
                let rn = (y_f64[i] - y_naive[i]).norm() / denom;
                if rs.is_finite() {
                    max_rel_scaled = max_rel_scaled.max(rs);
                }
                if rn.is_finite() {
                    max_rel_naive = max_rel_naive.max(rn);
                }
            }
            (
                max_rel_scaled,
                max_rel_naive,
                y_f64.to_vec(),
                y_scaled.to_vec(),
                y_naive.to_vec(),
            )
        }

        // --- Case A: in-band magnitude ~ 1.0. Both paths should be
        // f32-precise; this is the no-regression sanity check.
        let (rel_scaled_norm, rel_naive_norm, _, _, _) = measure(m, n, 1.0);
        assert!(
            rel_scaled_norm < 1e-5,
            "[normal-mag] scaled F32 rel = {rel_scaled_norm:.3e} should be ≤ 1e-5"
        );
        assert!(
            rel_naive_norm < 1e-5,
            "[normal-mag] naive F32 rel = {rel_naive_norm:.3e} should be ≤ 1e-5 \
             (sanity: in-band magnitudes are fine without scaling)"
        );

        // --- Case B: subnormal-territory magnitude. Naive downcast lands
        // entries below f32::MIN_POSITIVE_NORMAL ≈ 1.18e-38; scaled path
        // stores them near 1.0 with full f32 precision.
        let mag_sub = 1.0e-40_f64;
        let (rel_scaled_sub, rel_naive_sub, _, _, _) = measure(m, n, mag_sub);
        assert!(
            rel_scaled_sub < 1e-5,
            "[subnormal] scaled F32 rel = {rel_scaled_sub:.3e} should be ≤ 1e-5 \
             (Higham scaling brings entries into the f32 normal range)"
        );
        assert!(
            rel_naive_sub > rel_scaled_sub * 10.0,
            "[subnormal] naive F32 rel = {rel_naive_sub:.3e} should be \
             ≥ 10× worse than scaled = {rel_scaled_sub:.3e} (subnormal \
             entries lose precision under plain downcast)"
        );

        // --- Case C: overflow-territory magnitude. Naive downcast turns
        // every entry into ±Inf and the matvec produces NaN/Inf. Scaled
        // path stores the magnitude in the f64 scale field and remains
        // fully finite.
        let mag_over = 1.0e50_f64;
        let (_rel_scaled_over, _rel_naive_over, y_f64_over, y_scaled_over, y_naive_over) =
            measure(m, n, mag_over);
        let scaled_finite = y_scaled_over
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite());
        let naive_any_nonfinite = y_naive_over
            .iter()
            .any(|c| !c.re.is_finite() || !c.im.is_finite());
        assert!(
            scaled_finite,
            "[overflow] scaled F32 must produce finite output for |·| = 1e50, \
             got {y_scaled_over:?}"
        );
        assert!(
            naive_any_nonfinite,
            "[overflow] naive F32 must overflow for |·| = 1e50, but matvec is finite — \
             test fixture is wrong; expected at least one ±Inf or NaN entry"
        );
        // And the scaled output reproduces the f64 reference to f32 precision.
        let mut over_rel = 0.0_f64;
        for i in 0..m {
            let denom = y_f64_over[i].norm().max(f64::MIN_POSITIVE);
            let r = (y_f64_over[i] - y_scaled_over[i]).norm() / denom;
            if r.is_finite() {
                over_rel = over_rel.max(r);
            }
        }
        assert!(
            over_rel < 1e-5,
            "[overflow] scaled F32 rel vs f64 = {over_rel:.3e} should be ≤ 1e-5 even \
             at |·| = 1e50 (the f64 scale carries the magnitude)"
        );

        // --- Case D: scale-factor invariants from the factory function.
        // Build a fixture where U-magnitude = 5.0 and V-magnitude = 0.7
        // and verify the factored block holds those scales exactly.
        let mut u = DenseMatrix::<Complex64>::zeros(m, rank);
        let mut v = DenseMatrix::<Complex64>::zeros(n, rank);
        for i in 0..m {
            u[[i, 0]] = Complex64::new(5.0 * 0.6, 5.0 * 0.8); // |·| = 5
        }
        for j in 0..n {
            v[[j, 0]] = Complex64::new(0.7 * 0.6, 0.7 * 0.8); // |·| = 0.7
        }
        let factors = downcast_factors_f64_to_f32(u, v);
        match &factors {
            LowRankFactors::F32 {
                scale_u,
                scale_v,
                u,
                v,
            } => {
                assert!(
                    (scale_u - 5.0).abs() < 1e-12,
                    "scale_u = {scale_u} should equal max |U| = 5.0"
                );
                assert!(
                    (scale_v - 0.7).abs() < 1e-12,
                    "scale_v = {scale_v} should equal max |V| = 0.7"
                );
                let max_u_stored = u
                    .as_slice()
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f64)
                    .fold(0.0_f64, f64::max);
                let max_v_stored = v
                    .as_slice()
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f64)
                    .fold(0.0_f64, f64::max);
                assert!(
                    (max_u_stored - 1.0).abs() < 1e-6,
                    "max stored u_f32 magnitude = {max_u_stored:.3e} should be ≈ 1.0"
                );
                assert!(
                    (max_v_stored - 1.0).abs() < 1e-6,
                    "max stored v_f32 magnitude = {max_v_stored:.3e} should be ≈ 1.0"
                );
            }
            _ => unreachable!(),
        }

        // --- Case E: all-zero block edge case. Scale must floor at
        // MIN_SCALE_F32 so the inverse used by `downcast` stays finite;
        // reconstruction is exactly 0.
        let z_u = DenseMatrix::<Complex64>::zeros(4, 2);
        let z_v = DenseMatrix::<Complex64>::zeros(3, 2);
        let zero = downcast_factors_f64_to_f32(z_u, z_v);
        match &zero {
            LowRankFactors::F32 {
                scale_u, scale_v, ..
            } => {
                assert_eq!(*scale_u, MIN_SCALE_F32);
                assert_eq!(*scale_v, MIN_SCALE_F32);
            }
            _ => unreachable!(),
        }
        let zero_x = vec![Complex64::new(1.0, 0.0); 3];
        let zero_y = zero.apply(&zero_x);
        for entry in zero_y.iter() {
            assert_eq!(*entry, Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_hmatrix_matvec() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ];

        let y = hmat.matvec(x.as_slice());
        assert_eq!(y.len(), 4);

        // y should be non-zero for non-trivial x
        let y_norm: f64 = y.iter().map(|v| v.norm()).sum();
        assert!(y_norm > 1e-15, "Matvec result should be non-zero");
    }

    #[test]
    fn checked_hmatrix_operator_rejects_bad_vectors() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        assert!(matches!(
            hmat.matvec_checked(&[Complex64::new(0.0, 0.0); 3]),
            Err(HMatrixOperatorError::InputLength {
                expected: 4,
                actual: 3,
            })
        ));

        let mut nonfinite = vec![Complex64::new(0.0, 0.0); 4];
        nonfinite[2].im = f64::NAN;
        assert_eq!(
            hmat.matvec_checked(&nonfinite),
            Err(HMatrixOperatorError::NonFiniteValue { index: 2 })
        );

        let valid = vec![Complex64::new(1.0, 0.0); 4];
        let mut output = vec![Complex64::new(0.0, 0.0); 3];
        assert_eq!(
            hmat.matvec_into_checked(&valid, &mut output),
            Err(HMatrixOperatorError::OutputLength {
                expected: 4,
                actual: 3,
            })
        );

        assert!(hmat.matvec_transpose_checked(&valid).is_ok());
        assert!(hmat.matvec_hermitian_checked(&valid).is_ok());
    }

    #[test]
    fn checked_hmatrix_block_operator_matches_columns_and_rejects_bad_shapes() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);
        let input = DenseMatrix::from_row_major_vec(
            4,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-0.5, 0.25),
                Complex64::new(0.75, -0.5),
                Complex64::new(0.5, 0.25),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.25, 0.5),
                Complex64::new(0.0, -0.75),
            ],
        );

        let forward = hmat.matvec_block_checked(&input).unwrap();
        let transpose = hmat.matvec_transpose_block_checked(&input).unwrap();
        let hermitian = hmat.matvec_hermitian_block_checked(&input).unwrap();
        for rhs in 0..input.ncols() {
            let column: Vec<_> = (0..input.nrows()).map(|row| input[[row, rhs]]).collect();
            let expected_forward = hmat.matvec_checked(&column).unwrap();
            let expected_transpose = hmat.matvec_transpose_checked(&column).unwrap();
            let expected_hermitian = hmat.matvec_hermitian_checked(&column).unwrap();
            for row in 0..input.nrows() {
                assert!((forward[[row, rhs]] - expected_forward[row]).norm() < 1e-12);
                assert!((transpose[[row, rhs]] - expected_transpose[row]).norm() < 1e-12);
                assert!((hermitian[[row, rhs]] - expected_hermitian[row]).norm() < 1e-12);
            }
        }

        assert!(matches!(
            hmat.matvec_block_checked(&DenseMatrix::zeros(3, 2)),
            Err(HMatrixOperatorError::BlockInputRows {
                expected: 4,
                actual: 3,
            })
        ));
        let mut wrong_rows = DenseMatrix::zeros(3, 2);
        assert!(matches!(
            hmat.matvec_block_into_checked(&input, &mut wrong_rows),
            Err(HMatrixOperatorError::BlockOutputRows {
                expected: 4,
                actual: 3,
            })
        ));
        let mut wrong_columns = DenseMatrix::zeros(4, 1);
        assert!(matches!(
            hmat.matvec_transpose_block_into_checked(&input, &mut wrong_columns),
            Err(HMatrixOperatorError::BlockColumnCount {
                input: 2,
                output: 1,
            })
        ));
        let mut nonfinite = input.clone();
        nonfinite[[1, 1]].im = f64::NAN;
        assert!(matches!(
            hmat.matvec_hermitian_block_checked(&nonfinite),
            Err(HMatrixOperatorError::NonFiniteValue { index: 3 })
        ));
    }

    #[test]
    fn hmatrix_relative_action_error_checks_all_actions_and_rejects_bad_reference() {
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let (dense, hmat, n) = make_dense_and_hmat(&physics);
        let probes = vec![
            probe_vector(n),
            (0..n)
                .map(|index| {
                    let phase = 0.27 * (index + 1) as f64;
                    Complex64::new(phase.sin(), -phase.cos())
                })
                .collect(),
        ];

        let forward_references: Vec<Vec<Complex64>> =
            probes.iter().map(|probe| dense.matvec(probe)).collect();
        let mut transpose_references = Vec::with_capacity(probes.len());
        let mut hermitian_references = Vec::with_capacity(probes.len());
        for probe in &probes {
            let mut transpose = vec![Complex64::new(0.0, 0.0); n];
            dense.matvec_transpose_into(probe, &mut transpose);
            transpose_references.push(transpose);

            let mut hermitian = vec![Complex64::new(0.0, 0.0); n];
            dense.matvec_hermitian_into(probe, &mut hermitian);
            hermitian_references.push(hermitian);
        }

        let forward_error = hmat
            .relative_action_error(HMatrixAction::Forward, &probes, &forward_references)
            .expect("forward reference comparison");
        let transpose_error = hmat
            .relative_action_error(HMatrixAction::Transpose, &probes, &transpose_references)
            .expect("transpose reference comparison");
        let hermitian_error = hmat
            .relative_action_error(HMatrixAction::Hermitian, &probes, &hermitian_references)
            .expect("Hermitian reference comparison");
        assert!(forward_error < 1e-8, "forward error={forward_error:e}");
        assert!(
            transpose_error < 1e-8,
            "transpose error={transpose_error:e}"
        );
        assert!(
            hermitian_error < 1e-8,
            "Hermitian error={hermitian_error:e}"
        );

        assert_eq!(
            hmat.relative_action_error(HMatrixAction::Forward, &[], &[]),
            Err(HMatrixAccuracyError::EmptyProbeSet)
        );
        assert_eq!(
            hmat.relative_action_error(HMatrixAction::Forward, &probes, &[]),
            Err(HMatrixAccuracyError::ProbeCountMismatch {
                probes: probes.len(),
                references: 0,
            })
        );

        let mut bad_length = forward_references.clone();
        bad_length[0].pop();
        assert!(matches!(
            hmat.relative_action_error(HMatrixAction::Forward, &probes, &bad_length),
            Err(HMatrixAccuracyError::ReferenceLength { probe: 0, .. })
        ));

        let mut nonfinite_reference = forward_references.clone();
        nonfinite_reference[0][0].re = f64::NAN;
        assert!(matches!(
            hmat.relative_action_error(HMatrixAction::Forward, &probes, &nonfinite_reference),
            Err(HMatrixAccuracyError::NonFiniteReference { probe: 0, index: 0 })
        ));

        let mut nonfinite_probe = probes.clone();
        nonfinite_probe[0][0].im = f64::NAN;
        assert!(matches!(
            hmat.relative_action_error(
                HMatrixAction::Forward,
                &nonfinite_probe,
                &forward_references
            ),
            Err(HMatrixAccuracyError::Operator(
                HMatrixOperatorError::NonFiniteValue { index: 0 }
            ))
        ));
    }

    #[test]
    fn test_hmatrix_linear_operator() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        assert_eq!(hmat.num_rows(), 4);
        assert_eq!(hmat.num_cols(), 4);
        assert!(hmat.is_square());

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ];

        let y1 = hmat.apply(x.as_slice());
        let y2 = hmat.matvec(x.as_slice());

        // apply and matvec should produce the same result
        for i in 0..4 {
            assert!(
                (y1[i] - y2[i]).norm() < 1e-15,
                "apply and matvec should agree"
            );
        }

        // Test apply_transpose produces a result
        let yt = hmat.apply_transpose(x.as_slice());
        assert_eq!(yt.len(), 4);
    }

    #[test]
    fn test_hmatrix_apply_block_matches_column_matvec_for_both_precisions() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let x = DenseMatrix::from_fn(4, 3, |i, rhs| {
            Complex64::new(
                (0.31 * (i + 1 + rhs * 3) as f64).sin(),
                (0.17 * (i + 2 * rhs + 1) as f64).cos(),
            )
        });

        let compare = |hmat: &HMatrix, tolerance: f64| {
            assert!(!hmat.low_rank_blocks.is_empty());
            let block = hmat.matvec_block(&x);
            let mut trait_block = DenseMatrix::<Complex64>::zeros(4, 3);
            <HMatrix as LinearOperator<Complex64>>::apply_block(hmat, &x, &mut trait_block);

            for rhs in 0..x.ncols() {
                let expected = hmat.matvec(&x.column_to_vec(rhs));
                for i in 0..4 {
                    assert!(
                        (block[[i, rhs]] - expected[i]).norm() < tolerance,
                        "block matvec mismatch at row={i}, rhs={rhs}: block={:?}, expected={:?}",
                        block[[i, rhs]],
                        expected[i]
                    );
                    assert!(
                        (trait_block[[i, rhs]] - expected[i]).norm() < tolerance,
                        "trait block matvec mismatch at row={i}, rhs={rhs}: block={:?}, expected={:?}",
                        trait_block[[i, rhs]],
                        expected[i]
                    );
                }
            }
        };

        let hmat_f64 = build_hmatrix_with_variant_and_precision(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-10,
            16,
            AcaVariant::Hca,
            LowRankPrecision::F64,
        );
        compare(&hmat_f64, 1e-12);

        let hmat_f32 = build_hmatrix_with_variant_and_precision(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-10,
            16,
            AcaVariant::Hca,
            LowRankPrecision::F32,
        );
        compare(&hmat_f32, 1e-5);
    }

    /// Build the dense 4×4 TBEM reference for the standard test fixture,
    /// paired with its H-matrix compression. Returns `(dense, hmat)`; the
    /// H-matrix is asserted to contain at least one low-rank block so the
    /// parity checks genuinely exercise the factored path.
    fn make_dense_and_hmat(physics: &PhysicsParams) -> (DenseMatrix<Complex64>, HMatrix, usize) {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);

        let n = elements
            .iter()
            .map(|e| e.dof_addresses.len())
            .sum::<usize>();
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        for (i, src) in elements.iter().enumerate() {
            for (j, fld) in elements.iter().enumerate() {
                dense[[i, j]] =
                    compute_matrix_entry(i, j, src, fld, &nodes, physics, gamma, tau, beta, sign);
            }
        }

        let hmat = build_hmatrix(&elements, &nodes, &clusters, physics, 1e-10, 16);
        assert!(
            !hmat.low_rank_blocks.is_empty(),
            "fixture must exercise at least one low-rank block"
        );
        (dense, hmat, n)
    }

    fn relative_vec_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        let diff = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let reference = expected
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-30);
        diff / reference
    }

    fn probe_vector(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|i| Complex64::new((0.37 * (i + 1) as f64).sin(), (0.19 * (i + 1) as f64).cos()))
            .collect()
    }

    #[test]
    fn test_hmatrix_transpose_block_matches_columns() {
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let (_, hmat, n) = make_dense_and_hmat(&physics);
        let input = DenseMatrix::from_row_major_vec(
            n,
            2,
            (0..n * 2)
                .map(|index| {
                    let phase = 0.11 * (index + 1) as f64;
                    Complex64::new(phase.cos(), -phase.sin())
                })
                .collect(),
        );
        let block_output = hmat.matvec_transpose_block(&input);
        for rhs in 0..input.ncols() {
            let column = (0..n).map(|row| input[[row, rhs]]).collect::<Vec<_>>();
            let expected = hmat.apply_transpose(&column);
            for row in 0..n {
                assert!((block_output[[row, rhs]] - expected[row]).norm() < 1e-10);
            }
        }
    }

    /// Transpose parity vs dense: `matvec_transpose_into` must reproduce the
    /// dense `A^T x` reference (never previously validated — the old test
    /// only checked the output length).
    #[test]
    fn test_hmatrix_transpose_dense_parity() {
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let (dense, hmat, n) = make_dense_and_hmat(&physics);
        let x = probe_vector(n);

        let mut y_dense = vec![Complex64::new(0.0, 0.0); n];
        dense.matvec_transpose_into(&x, &mut y_dense);
        let y_hmat = hmat.apply_transpose(&x);
        let rel = relative_vec_error(&y_hmat, &y_dense);
        assert!(
            rel < 1e-6,
            "H-matrix transpose must match dense transpose (rel = {rel:.3e})"
        );

        // Also through matvec_transpose_into directly.
        let mut y_into = vec![Complex64::new(0.0, 0.0); n];
        hmat.matvec_transpose_into(&x, &mut y_into);
        let rel_into = relative_vec_error(&y_into, &y_dense);
        assert!(
            rel_into < 1e-6,
            "matvec_transpose_into must match dense transpose (rel = {rel_into:.3e})"
        );
    }

    /// Hermitian parity vs dense conj-transpose. Pins the conjugation
    /// semantics: storage is `A[i,j] = Σ_l U[i,l]·V[j,l]` (V^T, not V^H),
    /// so `A^H x` must conjugate **both** factors. A `V · (U^H · x)`
    /// implementation (conj on U only) fails this test on the complex
    /// Helmholtz kernel.
    #[test]
    fn test_hmatrix_hermitian_dense_parity() {
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let (dense, hmat, n) = make_dense_and_hmat(&physics);
        let x = probe_vector(n);

        let mut y_dense_h = vec![Complex64::new(0.0, 0.0); n];
        dense.matvec_hermitian_into(&x, &mut y_dense_h);
        let y_hmat_h = hmat.matvec_hermitian(&x);
        let rel = relative_vec_error(&y_hmat_h, &y_dense_h);
        assert!(
            rel < 1e-6,
            "H-matrix hermitian must match dense conj-transpose (rel = {rel:.3e})"
        );

        // Algebraic identity: A^H x = conj(A^T conj(x)) — cross-check the
        // hermitian kernel against the transpose kernel.
        let x_conj: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
        let mut y_via_transpose = hmat.apply_transpose(&x_conj);
        for v in y_via_transpose.iter_mut() {
            *v = v.conj();
        }
        let rel_xt = relative_vec_error(&y_hmat_h, &y_via_transpose);
        assert!(
            rel_xt < 1e-12,
            "hermitian must equal conj(transpose(conj(x))) (rel = {rel_xt:.3e})"
        );

        // Trait-path parity: LinearOperator::apply_hermitian must route to
        // the same kernel (not the conj+transpose default).
        let y_trait = <HMatrix as LinearOperator<Complex64>>::apply_hermitian(&hmat, &x);
        let rel_trait = relative_vec_error(&y_trait, &y_dense_h);
        assert!(
            rel_trait < 1e-6,
            "LinearOperator::apply_hermitian must match dense conj-transpose \
             (rel = {rel_trait:.3e})"
        );
    }

    /// Hermitian parity for the `LowRankPrecision::F32` storage path —
    /// conjugation happens on upcast f64 values, scales are real.
    #[test]
    fn test_hmatrix_hermitian_dense_parity_f32_storage() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let (dense, h_f64, n) = make_dense_and_hmat(&physics);
        let h_f32 = build_hmatrix_with_variant_and_precision(
            &elements,
            &nodes,
            &clusters,
            &physics,
            1e-10,
            16,
            AcaVariant::default(),
            LowRankPrecision::F32,
        );

        let x = probe_vector(n);
        let mut y_dense_h = vec![Complex64::new(0.0, 0.0); n];
        dense.matvec_hermitian_into(&x, &mut y_dense_h);
        let y_f64 = h_f64.matvec_hermitian(&x);
        let y_f32 = h_f32.matvec_hermitian(&x);

        let rel_f64 = relative_vec_error(&y_f64, &y_dense_h);
        let rel_f32 = relative_vec_error(&y_f32, &y_dense_h);
        assert!(
            rel_f64 < 1e-6,
            "F64 hermitian must match dense to round-off (rel = {rel_f64:.3e})"
        );
        assert!(
            rel_f32 < 1e-5,
            "F32 hermitian must stay inside the 1e-5 band (rel = {rel_f32:.3e})"
        );
    }

    /// LowRankFactors-level hermitian audit on synthetic factors with
    /// non-aligned phases: checks `apply_hermitian` against an explicitly
    /// materialised conj(U V^T) for both storage precisions.
    #[test]
    fn test_lowrank_factors_apply_hermitian_synthetic() {
        let m = 5;
        let n = 4;
        let rank = 3;
        let mut u = DenseMatrix::<Complex64>::zeros(m, rank);
        let mut v = DenseMatrix::<Complex64>::zeros(n, rank);
        for i in 0..m {
            for l in 0..rank {
                let phase = 0.29 * (i * rank + l + 1) as f64;
                u[[i, l]] = Complex64::new(phase.cos(), phase.sin());
            }
        }
        for j in 0..n {
            for l in 0..rank {
                let phase = 0.53 * (j * rank + l + 1) as f64;
                v[[j, l]] = Complex64::new(1.3 * phase.cos(), 1.3 * phase.sin());
            }
        }
        // Explicit block A = U V^T and its Hermitian.
        let mut ah = DenseMatrix::<Complex64>::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                let mut a_ij = Complex64::new(0.0, 0.0);
                for l in 0..rank {
                    a_ij += u[[i, l]] * v[[j, l]];
                }
                ah[[j, i]] = a_ij.conj();
            }
        }
        let x = probe_vector(m);
        let expected = ah.matvec(&x);

        let f64_factors = LowRankFactors::F64 {
            u: u.clone(),
            v: v.clone(),
        };
        let y_f64 = f64_factors.apply_hermitian(&x);
        let rel_f64 = relative_vec_error(&y_f64, &expected);
        assert!(
            rel_f64 < 1e-12,
            "F64 apply_hermitian vs explicit conj(U V^T): rel = {rel_f64:.3e}"
        );

        let f32_factors = downcast_factors_f64_to_f32(u, v);
        let y_f32 = f32_factors.apply_hermitian(&x);
        let rel_f32 = relative_vec_error(&y_f32, &expected);
        assert!(
            rel_f32 < 1e-5,
            "F32 apply_hermitian vs explicit conj(U V^T): rel = {rel_f32:.3e}"
        );
    }

    #[test]
    fn test_lowrank_factors_transpose_block_matches_columns() {
        let mut u = DenseMatrix::<Complex64>::zeros(5, 3);
        let mut v = DenseMatrix::<Complex64>::zeros(4, 3);
        for i in 0..u.nrows() {
            for l in 0..u.ncols() {
                let phase = 0.17 * (i * u.ncols() + l + 1) as f64;
                u[[i, l]] = Complex64::new(phase.cos(), phase.sin());
            }
        }
        for j in 0..v.nrows() {
            for l in 0..v.ncols() {
                let phase = 0.31 * (j * v.ncols() + l + 1) as f64;
                v[[j, l]] = Complex64::new(1.2 * phase.cos(), -0.8 * phase.sin());
            }
        }
        let input = DenseMatrix::from_row_major_vec(
            u.nrows(),
            2,
            (0..u.nrows() * 2)
                .map(|index| {
                    let phase = 0.23 * (index + 1) as f64;
                    Complex64::new(phase.cos(), phase.sin())
                })
                .collect(),
        );

        let expected_rows = v.nrows();
        for factors in [
            LowRankFactors::F64 {
                u: u.clone(),
                v: v.clone(),
            },
            downcast_factors_f64_to_f32(u, v),
        ] {
            let mut actual = DenseMatrix::<Complex64>::zeros(expected_rows, input.ncols());
            factors.apply_transpose_block_into(&input, &mut actual);
            for rhs in 0..input.ncols() {
                let column = (0..input.nrows())
                    .map(|row| input[[row, rhs]])
                    .collect::<Vec<_>>();
                let expected = factors.apply_transpose(&column);
                for row in 0..actual.nrows() {
                    assert!((actual[[row, rhs]] - expected[row]).norm() < 2e-5);
                }
            }
        }
    }

    /// Thread-count determinism for the transpose and hermitian schedules:
    /// 1-thread and 4-thread builds/apply must agree to round-off. This
    /// guards the field-cluster partition scatter against cross-partition
    /// output overlap.
    #[test]
    fn test_hmatrix_transpose_hermitian_thread_determinism() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);

        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool");
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-thread pool");

        let h1 =
            one_thread.install(|| build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-8, 16));
        let h4 = four_threads
            .install(|| build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-8, 16));

        assert_eq!(h1.partitions.len(), h1.transpose_partitions.len());
        assert_eq!(h4.partitions.len(), h4.transpose_partitions.len());

        let x = probe_vector(4);
        let yt1 = h1.apply_transpose(&x);
        let yt4 = h4.apply_transpose(&x);
        for (a, b) in yt1.iter().zip(yt4.iter()) {
            assert!(
                (a - b).norm() <= 1e-12,
                "thread-count-dependent H-matrix transpose: {a:?} vs {b:?}"
            );
        }

        let yh1 = h1.matvec_hermitian(&x);
        let yh4 = h4.matvec_hermitian(&x);
        for (a, b) in yh1.iter().zip(yh4.iter()) {
            assert!(
                (a - b).norm() <= 1e-12,
                "thread-count-dependent H-matrix hermitian: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn test_exact_stabilized_rejects_higher_order_layout() {
        let (mut elements, nodes) = make_test_mesh();
        elements[0].dof_addresses = vec![0, 1];
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);

        let error = apply_tbem_stabilized_exact(
            &elements,
            &nodes,
            &physics,
            1.0,
            1.0,
            &[Complex64::new(1.0, 0.0)],
        )
        .expect_err("exact reference must reject higher-order layouts");
        assert!(
            error.contains("constant-panel"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_exact_stabilized_hermitian_rejects_higher_order_layout() {
        let (mut elements, nodes) = make_test_mesh();
        elements[0].dof_addresses = vec![0, 1];
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);

        let error = apply_tbem_stabilized_exact_hermitian(
            &elements,
            &nodes,
            &physics,
            1.0,
            1.0,
            &[Complex64::new(1.0, 0.0)],
        )
        .expect_err("exact Hermitian reference must reject higher-order layouts");
        assert!(
            error.contains("constant-panel"),
            "unexpected error: {error}"
        );
    }

    /// Exact hermitian apply vs dense reference on the standard fixture:
    /// `apply_tbem_stabilized_exact_hermitian` must reproduce conj(Aᵀ)·x of
    /// the explicitly assembled dense operator.
    #[test]
    fn test_apply_tbem_stabilized_exact_hermitian_parity() {
        let (elements, nodes) = make_test_mesh();
        let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);
        let n = elements
            .iter()
            .map(|e| e.dof_addresses.len())
            .sum::<usize>();
        let x = probe_vector(n);

        let forward = apply_tbem_stabilized_exact(&elements, &nodes, &physics, 1.0, 1.0, &x)
            .expect("exact forward apply");
        let hermitian =
            apply_tbem_stabilized_exact_hermitian(&elements, &nodes, &physics, 1.0, 1.0, &x)
                .expect("exact hermitian apply");

        // Dense reference from compute_matrix_entry.
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta_scaled(1.0);
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        for (i, src) in elements.iter().enumerate() {
            for (j, fld) in elements.iter().enumerate() {
                dense[[i, j]] =
                    compute_matrix_entry(i, j, src, fld, &nodes, &physics, gamma, tau, beta, 1.0);
            }
        }
        let mut expected_forward = vec![Complex64::new(0.0, 0.0); n];
        dense.matvec_into(&x, &mut expected_forward);
        let mut expected_hermitian = vec![Complex64::new(0.0, 0.0); n];
        dense.matvec_hermitian_into(&x, &mut expected_hermitian);

        let rel_forward = relative_vec_error(&forward, &expected_forward);
        let rel_hermitian = relative_vec_error(&hermitian, &expected_hermitian);
        assert!(
            rel_forward < 1e-12,
            "exact forward apply vs dense: rel = {rel_forward:.3e}"
        );
        assert!(
            rel_hermitian < 1e-12,
            "exact hermitian apply vs dense conj-transpose: rel = {rel_hermitian:.3e}"
        );
    }

    /// Transpose partition coverage: the field-cluster schedule must cover
    /// every dense and low-rank block exactly once, and partition output
    /// DOF sets must be disjoint (no double-scatter).
    #[test]
    fn test_hmatrix_transpose_partitions_cover_blocks_once() {
        let (elements, nodes) = make_test_mesh();
        let clusters = make_test_clusters();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let hmat = build_hmatrix(&elements, &nodes, &clusters, &physics, 1e-4, 10);

        let dense_covered: usize = hmat
            .transpose_partitions
            .iter()
            .map(|p| p.dense_end - p.dense_start)
            .sum();
        assert_eq!(dense_covered, hmat.dense_blocks.len());
        let low_rank_covered: usize = hmat
            .transpose_partitions
            .iter()
            .map(|p| p.low_rank_end - p.low_rank_start)
            .sum();
        assert_eq!(low_rank_covered, hmat.low_rank_blocks.len());

        // Every cluster appears in exactly one transpose partition, so
        // output DOFs are disjoint across partitions.
        let mut cluster_owners: Vec<usize> = hmat
            .transpose_partitions
            .iter()
            .flat_map(|p| p.cluster_indices.iter().copied())
            .collect();
        cluster_owners.sort_unstable();
        let mut expected: Vec<usize> = (0..hmat.cluster_dof_indices.len()).collect();
        expected.sort_unstable();
        assert_eq!(cluster_owners, expected);
    }

    #[test]
    fn hmatrix_velocity_operator_matches_dense_rhs_and_exercises_far_field() {
        let (elements, nodes) = make_compression_test_mesh();
        let clusters = make_compression_test_clusters();
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta = physics.burton_miller_beta_scaled(1.0);
        let operator = HMatrixVelocityOperator::build(
            &elements, &nodes, &clusters, &physics, 1.0e-4, 16, sign, beta,
        )
        .expect("H-matrix velocity operator");
        assert!(
            operator
                .pressure_matrix()
                .low_rank_blocks
                .iter()
                .any(|block| block.source_cluster != block.field_cluster)
        );

        let velocity: Vec<Complex64> = (0..elements.len())
            .map(|index| {
                Complex64::new(
                    0.15 + 0.07 * index as f64,
                    -0.2 + 0.03 * (index * index) as f64,
                )
            })
            .collect();
        let mut prescribed = elements.clone();
        for (element, &value) in prescribed.iter_mut().zip(&velocity) {
            element.boundary_condition = BoundaryCondition::Velocity(vec![value]);
        }
        let expected =
            build_tbem_rhs_stabilized_with_beta(&prescribed, &nodes, &physics, sign, beta);
        let actual = operator.apply(&velocity).expect("compressed velocity RHS");
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let reference = expected
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-30);
        assert!(
            error / reference < 1.0e-7,
            "velocity RHS relative error: {error:e}"
        );
    }

    #[test]
    fn hmatrix_velocity_operator_block_transpose_and_hermitian_contracts() {
        let (elements, nodes) = make_compression_test_mesh();
        let clusters = make_compression_test_clusters();
        let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);
        let sign = -1.0;
        let beta = physics.burton_miller_beta_scaled(1.7);
        let operator = HMatrixVelocityOperator::build(
            &elements, &nodes, &clusters, &physics, 1.0e-4, 16, sign, beta,
        )
        .expect("H-matrix velocity operator");
        let a: Vec<Complex64> = (0..elements.len())
            .map(|index| Complex64::new(0.2 * index as f64, -0.1 - index as f64 * 0.04))
            .collect();
        let b: Vec<Complex64> = (0..elements.len())
            .map(|index| Complex64::new(-0.3 + index as f64 * 0.05, 0.07 * index as f64))
            .collect();
        let qb = operator.apply(&b).expect("forward apply");
        let qta = operator.apply_transpose(&a).expect("transpose apply");
        let qha = operator.apply_hermitian(&a).expect("Hermitian apply");
        let transpose_lhs = a.iter().zip(&qb).map(|(x, y)| *x * *y).sum::<Complex64>();
        let transpose_rhs = qta.iter().zip(&b).map(|(x, y)| *x * *y).sum::<Complex64>();
        assert!((transpose_lhs - transpose_rhs).norm() < 1.0e-8);
        let hermitian_lhs = a
            .iter()
            .zip(&qb)
            .map(|(x, y)| x.conj() * *y)
            .sum::<Complex64>();
        let hermitian_rhs = qha
            .iter()
            .zip(&b)
            .map(|(x, y)| x.conj() * *y)
            .sum::<Complex64>();
        assert!((hermitian_lhs - hermitian_rhs).norm() < 1.0e-8);

        let block_input = DenseMatrix::from_row_major_vec(
            elements.len(),
            2,
            a.iter().zip(&b).flat_map(|(x, y)| [*x, *y]).collect(),
        );
        let block = operator.apply_block(&block_input).expect("block apply");
        let qa = operator.apply(&a).expect("first block");
        for row in 0..elements.len() {
            assert!((block[[row, 0]] - qa[row]).norm() < 1.0e-10);
            assert!((block[[row, 1]] - qb[row]).norm() < 1.0e-10);
        }
    }

    #[test]
    fn hmatrix_velocity_operator_rejects_malformed_inputs_and_build_parameters() {
        let (elements, nodes) = make_compression_test_mesh();
        let clusters = make_compression_test_clusters();
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta = physics.burton_miller_beta_scaled(1.0);
        let operator = HMatrixVelocityOperator::build(
            &elements, &nodes, &clusters, &physics, 1.0e-10, 16, sign, beta,
        )
        .expect("H-matrix velocity operator");
        let velocity = vec![Complex64::new(0.0, 0.0); elements.len()];
        assert!(matches!(
            operator.apply(&velocity[..velocity.len() - 1]),
            Err(TbemVelocityOperatorError::VelocityLength { .. })
        ));
        let mut non_finite = velocity.clone();
        non_finite[3] = Complex64::new(f64::NAN, 0.0);
        assert!(matches!(
            operator.apply(&non_finite),
            Err(TbemVelocityOperatorError::NonFiniteVelocity { index: 3 })
        ));
        assert!(matches!(
            HMatrixVelocityOperator::build(
                &elements, &nodes, &clusters, &physics, 0.0, 16, sign, beta,
            ),
            Err(TbemVelocityOperatorError::InvalidParameter {
                name: "aca_tolerance"
            })
        ));
        assert!(matches!(
            HMatrixVelocityOperator::build(
                &elements, &nodes, &clusters, &physics, 1.0e-10, 0, sign, beta,
            ),
            Err(TbemVelocityOperatorError::InvalidParameter { name: "max_rank" })
        ));
        assert!(matches!(
            HMatrixVelocityOperator::build(
                &elements,
                &nodes,
                &[],
                &physics,
                1.0e-10,
                16,
                sign,
                beta,
            ),
            Err(TbemVelocityOperatorError::InvalidParameter { name: "clusters" })
        ));
    }
}
