//! Single-Level Fast Multipole Method (SLFMM) assembly
//!
//! The SLFMM decomposes the BEM interaction matrix as:
//! ```text
//! [A] = [N] + [S][D][T]
//! ```
//! where:
//! - \[N\] is the near-field matrix (direct interactions between nearby elements)
//! - \[T\] is the T-matrix (element-to-cluster multipole expansion)
//! - \[D\] is the D-matrix (cluster-to-cluster translation)
//! - \[S\] is the S-matrix (cluster-to-element local expansion)
//!
//! **Note**: This module requires either the `native` or `wasm` feature for parallel processing.
//! Parallelism is provided via rayon (native) or wasm-bindgen-rayon (WASM).

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;

use crate::core::parallel::{
    BEM_BLOCKS_PER_THREAD, gated_enumerate_filter_map, gated_enumerate_map, gated_flat_map,
    gated_map,
};

use crate::core::assembly::common::{
    FmmAccuracyError, FmmAction, basis_order_for_proxy, checked_fmm_relative_action_error,
    compute_near_block_stabilized, compute_near_block_stabilized_with_basis, count_dofs,
    decode_dof_proxy_group, get_element_coords, proxy_local_index,
};
use crate::core::assembly::hmatrix::compute_matrix_entry_wavenumber_derivative_curved;
use crate::core::integration::{quad_quadrature, triangle_quadrature, unit_sphere_quadrature};
use crate::core::mesh::element::{basis_values, quadrature_geometry};
use crate::core::types::{BasisOrder, Cluster, Element, PhysicsParams};
use sonium_solvers::traits::LinearOperator;
use sonium_wave::special::{legendre_polynomials, spherical_hankel_first_kind};

/// Result of SLFMM assembly
#[derive(Clone)]
pub struct SlfmmSystem {
    /// Near-field coefficient matrix (sparse, stored as dense blocks)
    pub near_matrix: Vec<NearFieldBlock>,
    /// T-matrix for each cluster (element DOFs → multipole expansion)
    pub t_matrices: Vec<DenseMatrix<Complex64>>,
    /// T-vector for RHS contribution
    pub t_vector: Vec<Complex64>,
    /// D-matrix entries for far cluster pairs
    pub d_matrices: Vec<DMatrixEntry>,
    /// S-matrix for each cluster (multipole expansion → field DOFs)
    pub s_matrices: Vec<DenseMatrix<Complex64>>,
    /// RHS vector
    pub rhs: Vec<Complex64>,
    /// Number of DOFs
    pub num_dofs: usize,
    /// Number of integration points on unit sphere
    pub num_sphere_points: usize,
    /// Number of expansion terms
    pub num_expansion_terms: usize,
    /// Number of clusters
    pub num_clusters: usize,
    /// Cluster DOF mappings: for each cluster, the global DOF indices
    pub cluster_dof_indices: Vec<Vec<usize>>,
}

/// Checked local-row action failures for the single-level FMM shell.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SlfmmRowActionError {
    /// The input vector does not match the assembled operator dimension.
    #[error("SLFMM {operation} input length {actual} does not match expected {expected}")]
    Dimension {
        /// Action name.
        operation: &'static str,
        /// Supplied input length.
        actual: usize,
        /// Required input length.
        expected: usize,
    },
    /// A requested output row lies outside the assembled operator.
    #[error("SLFMM row {row} is outside operator dimension {num_dofs}")]
    InvalidRow {
        /// Requested global row.
        row: usize,
        /// Operator dimension.
        num_dofs: usize,
    },
    /// An input entry is not finite.
    #[error("SLFMM {operation} input entry {index} must be finite")]
    NonFiniteInput {
        /// Action name.
        operation: &'static str,
        /// Invalid input index.
        index: usize,
    },
    /// The computed action contains a non-finite value.
    #[error("SLFMM {operation} output entry {index} is non-finite")]
    NonFiniteOutput {
        /// Action name.
        operation: &'static str,
        /// Invalid output index.
        index: usize,
    },
    /// The assembled hierarchy does not contain the expected number of
    /// per-cluster tables or DOF mappings.
    #[error("SLFMM {operation} {stage} count {actual} does not match expected {expected}")]
    LayoutCount {
        /// Action name.
        operation: &'static str,
        /// Hierarchy component being checked.
        stage: &'static str,
        /// Actual number of entries.
        actual: usize,
        /// Required number of entries.
        expected: usize,
    },
    /// A hierarchy cluster references a cluster that is not present.
    #[error("SLFMM {operation} {side} cluster {cluster} outside hierarchy size {available}")]
    InvalidCluster {
        /// Action name.
        operation: &'static str,
        /// Cluster reference side.
        side: &'static str,
        /// Invalid cluster index.
        cluster: usize,
        /// Number of available clusters.
        available: usize,
    },
    /// A cluster mapping references a global DOF outside the operator.
    #[error("SLFMM {operation} cluster {cluster} DOF {local} maps to {global}, outside {num_dofs}")]
    InvalidDof {
        /// Action name.
        operation: &'static str,
        /// Cluster index.
        cluster: usize,
        /// Local cluster DOF slot.
        local: usize,
        /// Global DOF index.
        global: usize,
        /// Operator dimension.
        num_dofs: usize,
    },
    /// A near-field block shape does not match its cluster mappings.
    #[error(
        "SLFMM {operation} near block {block} shape {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    NearFieldShape {
        /// Action name.
        operation: &'static str,
        /// Near-field block index.
        block: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
    },
    /// A multipole/local expansion table has an invalid shape.
    #[error(
        "SLFMM {operation} {stage} table for cluster {cluster} shape {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    ExpansionShape {
        /// Action name.
        operation: &'static str,
        /// Expansion stage (`T` or `S`).
        stage: &'static str,
        /// Cluster index.
        cluster: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
    },
    /// A translation entry has an invalid diagonal length.
    #[error("SLFMM {operation} translation {entry} diagonal length {actual}, expected {expected}")]
    TranslationShape {
        /// Action name.
        operation: &'static str,
        /// Translation entry index.
        entry: usize,
        /// Actual diagonal length.
        actual: usize,
        /// Expected diagonal length.
        expected: usize,
    },
}

/// Near-field block between two clusters
#[derive(Debug, Clone)]
pub struct NearFieldBlock {
    /// Source cluster index
    pub source_cluster: usize,
    /// Field cluster index
    pub field_cluster: usize,
    /// Dense coefficient matrix for this block
    pub coefficients: DenseMatrix<Complex64>,
}

/// Validation failure while materializing an SLFMM near-field matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlfmmNearFieldError {
    /// A block refers to a cluster that is not present in the mapping.
    InvalidCluster {
        /// Block source/field side.
        side: &'static str,
        /// Invalid cluster index.
        cluster: usize,
        /// Number of mapped clusters.
        available: usize,
    },
    /// A block shape does not match its source/field DOF mappings.
    BlockShape {
        /// Near-field block index.
        block: usize,
        /// Expected source-row count.
        source_dofs: usize,
        /// Expected field-column count.
        field_dofs: usize,
        /// Actual matrix row count.
        rows: usize,
        /// Actual matrix column count.
        cols: usize,
    },
    /// A mapped global DOF lies outside the system matrix.
    GlobalDof {
        /// Near-field block index.
        block: usize,
        /// Source/field side.
        side: &'static str,
        /// Local DOF position.
        local: usize,
        /// Invalid global DOF index.
        global: usize,
        /// Matrix dimension.
        num_dofs: usize,
    },
    /// A coefficient is NaN or infinite.
    NonFiniteCoefficient {
        /// Near-field block index.
        block: usize,
        /// Matrix row.
        row: usize,
        /// Matrix column.
        col: usize,
    },
}

/// Validation failure while admitting a mesh to the constant-panel SLFMM
/// builder.
///
/// SLFMM's production multipole and near-field kernels currently use one
/// coefficient per element.  Higher-order or shared DOF layouts must be
/// rejected before quadrature starts rather than being silently interpreted
/// as constant panels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlfmmAssemblyError {
    /// The element has a basis layout other than one DOF per active panel.
    UnsupportedBasis {
        /// Element index.
        element: usize,
        /// Number of local coefficient slots.
        dofs: usize,
    },
    /// Element DOF count does not match the requested basis order.
    InvalidBasisDofCount {
        /// Element index.
        element: usize,
        /// Expected local basis count.
        expected: usize,
        /// Actual local DOF count.
        actual: usize,
    },
    /// An active element contains a global DOF outside the assembled vector.
    InvalidDofAddress {
        /// Element index.
        element: usize,
        /// Invalid global DOF.
        global: usize,
        /// Inferred system dimension.
        num_dofs: usize,
    },
    /// Two active elements share a global DOF, which is not supported by the
    /// constant-panel multipole incidence map.
    DuplicateGlobalDof {
        /// First element owning the global DOF.
        first_element: usize,
        /// Second element attempting to own the global DOF.
        second_element: usize,
        /// Shared global DOF.
        global: usize,
    },
    /// A cluster references an element that is not present in the mesh.
    InvalidClusterElement {
        /// Cluster index.
        cluster: usize,
        /// Referenced element index.
        element: usize,
        /// Number of mesh elements.
        available: usize,
    },
    /// An active element occurs in more than one cluster.
    DuplicateElementMembership {
        /// Element index.
        element: usize,
        /// First owning cluster.
        first_cluster: usize,
        /// Second owning cluster.
        second_cluster: usize,
    },
    /// An active element is absent from the cluster partition.
    MissingElement {
        /// Element index.
        element: usize,
    },
    /// A mesh connectivity entry points outside the coordinate array.
    InvalidNode {
        /// Element index.
        element: usize,
        /// Local connectivity slot.
        local: usize,
        /// Referenced node index.
        node: usize,
        /// Number of coordinate nodes.
        available: usize,
    },
    /// A quadrature/expansion dimension would produce an invalid sphere rule.
    InvalidQuadrature {
        /// Polar quadrature order.
        n_theta: usize,
        /// Azimuthal quadrature order.
        n_phi: usize,
        /// Multipole truncation order.
        n_terms: usize,
    },
}

impl std::fmt::Display for SlfmmAssemblyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBasis { element, dofs } => write!(
                f,
                "SLFMM element {element} has {dofs} DOFs; only constant one-DOF panels are supported"
            ),
            Self::InvalidBasisDofCount {
                element,
                expected,
                actual,
            } => write!(
                f,
                "SLFMM element {element} has {actual} DOFs, expected {expected} for the requested basis"
            ),
            Self::InvalidDofAddress {
                element,
                global,
                num_dofs,
            } => write!(
                f,
                "SLFMM element {element} DOF {global} is outside {num_dofs} assembled DOFs"
            ),
            Self::DuplicateGlobalDof {
                first_element,
                second_element,
                global,
            } => write!(
                f,
                "SLFMM global DOF {global} is shared by elements {first_element} and {second_element}"
            ),
            Self::InvalidClusterElement {
                cluster,
                element,
                available,
            } => write!(
                f,
                "SLFMM cluster {cluster} references element {element} (mesh has {available})"
            ),
            Self::DuplicateElementMembership {
                element,
                first_cluster,
                second_cluster,
            } => write!(
                f,
                "SLFMM active element {element} occurs in clusters {first_cluster} and {second_cluster}"
            ),
            Self::MissingElement { element } => {
                write!(
                    f,
                    "SLFMM active element {element} is absent from the cluster partition"
                )
            }
            Self::InvalidNode {
                element,
                local,
                node,
                available,
            } => write!(
                f,
                "SLFMM element {element} connectivity node {node} at local slot {local} is outside {available} nodes"
            ),
            Self::InvalidQuadrature {
                n_theta,
                n_phi,
                n_terms,
            } => write!(
                f,
                "SLFMM quadrature dimensions must be non-zero (theta={n_theta}, phi={n_phi}, terms={n_terms})"
            ),
        }
    }
}

impl std::error::Error for SlfmmAssemblyError {}

impl std::fmt::Display for SlfmmNearFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCluster {
                side,
                cluster,
                available,
            } => write!(
                f,
                "SLFMM near-field {side} cluster {cluster} is out of range (available {available})"
            ),
            Self::BlockShape {
                block,
                source_dofs,
                field_dofs,
                rows,
                cols,
            } => write!(
                f,
                "SLFMM near-field block {block} has {rows}x{cols} coefficients, expected {source_dofs}x{field_dofs}"
            ),
            Self::GlobalDof {
                block,
                side,
                local,
                global,
                num_dofs,
            } => write!(
                f,
                "SLFMM near-field block {block} {side} DOF {local} maps to {global}, outside {num_dofs} DOFs"
            ),
            Self::NonFiniteCoefficient { block, row, col } => write!(
                f,
                "SLFMM near-field block {block} coefficient ({row},{col}) is non-finite"
            ),
        }
    }
}

impl std::error::Error for SlfmmNearFieldError {}

/// D-matrix entry for a far cluster pair
///
/// The D-matrix for SLFMM is diagonal (same multipole index in source and field),
/// so we store only the diagonal entries to save memory.
/// This reduces storage from O(n²) to O(n) per cluster pair.
#[derive(Debug, Clone)]
pub struct DMatrixEntry {
    /// Source cluster index
    pub source_cluster: usize,
    /// Field cluster index
    pub field_cluster: usize,
    /// Diagonal translation coefficients (length = num_sphere_points)
    /// The full D-matrix would be: `D[p,q] = diagonal[p]` if p == q, else 0
    pub diagonal: Vec<Complex64>,
}

/// Analytic fixed-geometry wavenumber derivative of an SLFMM factorization.
///
/// The derivative is stored in the same near/T/D/S factorization as the value
/// operator.  [`Self::apply`] evaluates the product rule without materializing
/// a dense `N x N` derivative matrix:
///
/// `A' x = N' x + S' D T x + S D' T x + S D T' x`.
#[derive(Debug, Clone)]
pub struct SlfmmWavenumberDerivative {
    /// Near-field derivative blocks.
    pub near_matrix: Vec<NearFieldBlock>,
    /// Derivative of the element-to-cluster factors.
    pub t_matrices: Vec<DenseMatrix<Complex64>>,
    /// Derivative of the diagonal cluster translations.
    pub d_matrices: Vec<DMatrixEntry>,
    /// Derivative of the cluster-to-element factors.
    pub s_matrices: Vec<DenseMatrix<Complex64>>,
}

impl SlfmmWavenumberDerivative {
    /// Apply the factorized derivative to a global coefficient vector.
    pub fn apply(&self, base: &SlfmmSystem, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), base.num_dofs);
        assert_eq!(self.t_matrices.len(), base.t_matrices.len());
        assert_eq!(self.s_matrices.len(), base.s_matrices.len());
        assert_eq!(self.d_matrices.len(), base.d_matrices.len());

        let mut y = vec![Complex64::new(0.0, 0.0); base.num_dofs];
        for block in &self.near_matrix {
            let source_dofs = &base.cluster_dof_indices[block.source_cluster];
            let field_dofs = &base.cluster_dof_indices[block.field_cluster];
            let x_local: Vec<_> = field_dofs.iter().map(|&index| x[index]).collect();
            let contribution = block.coefficients.matvec(&x_local);
            for (local, &global) in source_dofs.iter().enumerate() {
                y[global] += contribution[local];
            }
        }

        let mut multipoles =
            vec![vec![Complex64::new(0.0, 0.0); base.num_sphere_points]; base.num_clusters];
        let mut derivative_multipoles = multipoles.clone();
        for cluster in 0..base.num_clusters {
            let dofs = &base.cluster_dof_indices[cluster];
            if dofs.is_empty() {
                continue;
            }
            let x_local: Vec<_> = dofs.iter().map(|&index| x[index]).collect();
            multipoles[cluster] = base.t_matrices[cluster].matvec(&x_local);
            derivative_multipoles[cluster] = self.t_matrices[cluster].matvec(&x_local);
        }

        let mut locals =
            vec![vec![Complex64::new(0.0, 0.0); base.num_sphere_points]; base.num_clusters];
        let mut derivative_locals = locals.clone();
        for (entry_index, entry) in base.d_matrices.iter().enumerate() {
            let derivative = &self.d_matrices[entry_index];
            let source = &multipoles[entry.source_cluster];
            let derivative_source = &derivative_multipoles[entry.source_cluster];
            let target = &mut locals[entry.field_cluster];
            let derivative_target = &mut derivative_locals[entry.field_cluster];
            for point in 0..base.num_sphere_points {
                target[point] += entry.diagonal[point] * source[point];
                derivative_target[point] += derivative.diagonal[point] * source[point]
                    + entry.diagonal[point] * derivative_source[point];
            }
        }

        for cluster in 0..base.num_clusters {
            let dofs = &base.cluster_dof_indices[cluster];
            if dofs.is_empty() {
                continue;
            }
            let derivative = self.s_matrices[cluster].matvec(&locals[cluster]);
            let derivative_value = base.s_matrices[cluster].matvec(&derivative_locals[cluster]);
            for (local, &global) in dofs.iter().enumerate() {
                y[global] += derivative[local] + derivative_value[local];
            }
        }
        y
    }
}

impl SlfmmSystem {
    /// Validate hierarchy metadata before entering the legacy infallible
    /// matrix-vector kernels.  Builders normally establish these invariants,
    /// but the public fields also permit deserializers and distributed owners
    /// to construct a malformed system after assembly.
    fn validate_action_layout(&self, operation: &'static str) -> Result<(), SlfmmRowActionError> {
        let expected = self.num_clusters;
        for (stage, actual) in [
            ("cluster DOF mappings", self.cluster_dof_indices.len()),
            ("T matrices", self.t_matrices.len()),
            ("S matrices", self.s_matrices.len()),
        ] {
            if actual != expected {
                return Err(SlfmmRowActionError::LayoutCount {
                    operation,
                    stage,
                    actual,
                    expected,
                });
            }
        }

        for cluster in 0..self.num_clusters {
            let dofs = &self.cluster_dof_indices[cluster];
            for (local, &global) in dofs.iter().enumerate() {
                if global >= self.num_dofs {
                    return Err(SlfmmRowActionError::InvalidDof {
                        operation,
                        cluster,
                        local,
                        global,
                        num_dofs: self.num_dofs,
                    });
                }
            }

            let t_matrix = &self.t_matrices[cluster];
            let s_matrix = &self.s_matrices[cluster];
            let t_shape_is_empty = dofs.is_empty() && t_matrix.as_slice().is_empty();
            let s_shape_is_empty = dofs.is_empty() && s_matrix.as_slice().is_empty();
            if !t_shape_is_empty
                && (t_matrix.nrows() != self.num_sphere_points || t_matrix.ncols() != dofs.len())
            {
                return Err(SlfmmRowActionError::ExpansionShape {
                    operation,
                    stage: "T",
                    cluster,
                    rows: t_matrix.nrows(),
                    cols: t_matrix.ncols(),
                    expected_rows: self.num_sphere_points,
                    expected_cols: dofs.len(),
                });
            }
            if !s_shape_is_empty
                && (s_matrix.nrows() != dofs.len() || s_matrix.ncols() != self.num_sphere_points)
            {
                return Err(SlfmmRowActionError::ExpansionShape {
                    operation,
                    stage: "S",
                    cluster,
                    rows: s_matrix.nrows(),
                    cols: s_matrix.ncols(),
                    expected_rows: dofs.len(),
                    expected_cols: self.num_sphere_points,
                });
            }
        }

        for (block, near) in self.near_matrix.iter().enumerate() {
            let source_dofs = self.cluster_dof_indices.get(near.source_cluster).ok_or(
                SlfmmRowActionError::InvalidCluster {
                    operation,
                    side: "source",
                    cluster: near.source_cluster,
                    available: self.cluster_dof_indices.len(),
                },
            )?;
            let field_dofs = self.cluster_dof_indices.get(near.field_cluster).ok_or(
                SlfmmRowActionError::InvalidCluster {
                    operation,
                    side: "field",
                    cluster: near.field_cluster,
                    available: self.cluster_dof_indices.len(),
                },
            )?;
            if near.coefficients.nrows() != source_dofs.len()
                || near.coefficients.ncols() != field_dofs.len()
            {
                return Err(SlfmmRowActionError::NearFieldShape {
                    operation,
                    block,
                    rows: near.coefficients.nrows(),
                    cols: near.coefficients.ncols(),
                    expected_rows: source_dofs.len(),
                    expected_cols: field_dofs.len(),
                });
            }
        }

        for (entry, translation) in self.d_matrices.iter().enumerate() {
            if translation.source_cluster >= self.num_clusters {
                return Err(SlfmmRowActionError::InvalidCluster {
                    operation,
                    side: "translation source",
                    cluster: translation.source_cluster,
                    available: self.num_clusters,
                });
            }
            if translation.field_cluster >= self.num_clusters {
                return Err(SlfmmRowActionError::InvalidCluster {
                    operation,
                    side: "translation field",
                    cluster: translation.field_cluster,
                    available: self.num_clusters,
                });
            }
            if translation.diagonal.len() != self.num_sphere_points {
                return Err(SlfmmRowActionError::TranslationShape {
                    operation,
                    entry,
                    actual: translation.diagonal.len(),
                    expected: self.num_sphere_points,
                });
            }
        }

        Ok(())
    }

    /// Checked forward SLFMM action that rejects malformed hierarchy state,
    /// invalid input, and non-finite output before returning to callers.
    pub fn matvec_checked(&self, x: &[Complex64]) -> Result<Vec<Complex64>, SlfmmRowActionError> {
        validate_row_action_input(x, self.num_dofs, "forward")?;
        self.validate_action_layout("forward")?;
        let output = self.matvec(x);
        validate_row_action_output(&output, "forward")?;
        Ok(output)
    }

    /// Checked algebraic-transpose SLFMM action.
    pub fn matvec_transpose_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, SlfmmRowActionError> {
        validate_row_action_input(x, self.num_dofs, "transpose")?;
        self.validate_action_layout("transpose")?;
        let output = self.matvec_transpose(x);
        validate_row_action_output(&output, "transpose")?;
        Ok(output)
    }

    /// Checked Hermitian-transpose SLFMM action.
    pub fn apply_hermitian_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, SlfmmRowActionError> {
        validate_row_action_input(x, self.num_dofs, "Hermitian")?;
        self.validate_action_layout("Hermitian")?;
        let output = self.apply_hermitian(x);
        validate_row_action_output(&output, "Hermitian")?;
        Ok(output)
    }

    /// Compare one SLFMM action against independently assembled reference
    /// outputs for deterministic probe vectors.
    pub fn relative_action_error(
        &self,
        action: FmmAction,
        probes: &[Vec<Complex64>],
        reference_outputs: &[Vec<Complex64>],
    ) -> Result<f64, FmmAccuracyError> {
        checked_fmm_relative_action_error(
            action,
            probes,
            reference_outputs,
            self.num_dofs,
            |action, probe| match action {
                FmmAction::Forward => self.matvec(probe),
                FmmAction::Transpose => self.matvec_transpose(probe),
                FmmAction::Hermitian => self.apply_hermitian(probe),
            },
        )
    }

    /// Create a new empty SLFMM system
    pub fn new(
        num_dofs: usize,
        num_clusters: usize,
        num_sphere_points: usize,
        num_expansion_terms: usize,
    ) -> Self {
        Self {
            near_matrix: Vec::new(),
            t_matrices: Vec::with_capacity(num_clusters),
            t_vector: vec![Complex64::new(0.0, 0.0); num_sphere_points * num_clusters],
            d_matrices: Vec::new(),
            s_matrices: Vec::with_capacity(num_clusters),
            rhs: vec![Complex64::new(0.0, 0.0); num_dofs],
            num_dofs,
            num_sphere_points,
            num_expansion_terms,
            num_clusters,
            cluster_dof_indices: Vec::with_capacity(num_clusters),
        }
    }

    /// Extract the near-field matrix as a dense matrix for preconditioning
    ///
    /// This assembles only the near-field blocks into a dense matrix,
    /// which is much smaller than the full matrix (O(N) vs O(N²) entries).
    pub fn extract_near_field_matrix(&self) -> DenseMatrix<Complex64> {
        let mut matrix = DenseMatrix::<Complex64>::zeros(self.num_dofs, self.num_dofs);

        for block in &self.near_matrix {
            let src_dofs = &self.cluster_dof_indices[block.source_cluster];
            let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

            // Copy block to global matrix
            // No symmetric duplication needed: build_near_field already emits
            // both (i,j) and (j,i) blocks via the symmetric near_clusters lists.
            for (local_i, &global_i) in src_dofs.iter().enumerate() {
                for (local_j, &global_j) in fld_dofs.iter().enumerate() {
                    matrix[[global_i, global_j]] += block.coefficients[[local_i, local_j]];
                }
            }
        }

        matrix
    }

    /// Checked near-field extraction for solver/preconditioner boundaries.
    ///
    /// Validates cluster references, block dimensions, global DOF mappings,
    /// and coefficient finiteness before indexing or materializing the dense
    /// matrix. The legacy [`Self::extract_near_field_matrix`] API remains
    /// available for trusted compatibility callers.
    pub fn try_extract_near_field_matrix(
        &self,
    ) -> Result<DenseMatrix<Complex64>, SlfmmNearFieldError> {
        let mut matrix = DenseMatrix::<Complex64>::zeros(self.num_dofs, self.num_dofs);

        for (block_index, block) in self.near_matrix.iter().enumerate() {
            let source_dofs = self.cluster_dof_indices.get(block.source_cluster).ok_or(
                SlfmmNearFieldError::InvalidCluster {
                    side: "source",
                    cluster: block.source_cluster,
                    available: self.cluster_dof_indices.len(),
                },
            )?;
            let field_dofs = self.cluster_dof_indices.get(block.field_cluster).ok_or(
                SlfmmNearFieldError::InvalidCluster {
                    side: "field",
                    cluster: block.field_cluster,
                    available: self.cluster_dof_indices.len(),
                },
            )?;

            if block.coefficients.nrows() != source_dofs.len()
                || block.coefficients.ncols() != field_dofs.len()
            {
                return Err(SlfmmNearFieldError::BlockShape {
                    block: block_index,
                    source_dofs: source_dofs.len(),
                    field_dofs: field_dofs.len(),
                    rows: block.coefficients.nrows(),
                    cols: block.coefficients.ncols(),
                });
            }

            for (local, &global) in source_dofs.iter().enumerate() {
                if global >= self.num_dofs {
                    return Err(SlfmmNearFieldError::GlobalDof {
                        block: block_index,
                        side: "source",
                        local,
                        global,
                        num_dofs: self.num_dofs,
                    });
                }
            }
            for (local, &global) in field_dofs.iter().enumerate() {
                if global >= self.num_dofs {
                    return Err(SlfmmNearFieldError::GlobalDof {
                        block: block_index,
                        side: "field",
                        local,
                        global,
                        num_dofs: self.num_dofs,
                    });
                }
            }

            for row in 0..block.coefficients.nrows() {
                for col in 0..block.coefficients.ncols() {
                    let value = block.coefficients[[row, col]];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(SlfmmNearFieldError::NonFiniteCoefficient {
                            block: block_index,
                            row,
                            col,
                        });
                    }
                    matrix[[source_dofs[row], field_dofs[col]]] += value;
                }
            }
        }

        Ok(matrix)
    }

    /// Apply only the near-field portion of the matvec: y = [N] * x.
    ///
    /// Mirrors the near-field branch of `matvec` so callers (e.g. QA bins)
    /// can compute the near-field-only contribution without touching the
    /// private `Array2` per-block storage.
    pub fn near_field_matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        for block in &self.near_matrix {
            let src_dofs = &self.cluster_dof_indices[block.source_cluster];
            let fld_dofs = &self.cluster_dof_indices[block.field_cluster];
            let x_local: Vec<Complex64> = fld_dofs.iter().map(|&i| x[i]).collect();
            let y_local = block.coefficients.matvec(&x_local);
            for (li, &gi) in src_dofs.iter().enumerate() {
                y[gi] += y_local[li];
            }
        }
        y
    }

    /// Apply the SLFMM operator: y = (\[N\] + \[S\]\[D\]\[T\]) * x
    ///
    /// This is used in iterative solvers (CGS, BiCGSTAB).
    ///
    /// The decomposition is:
    /// - Near-field: Direct element-to-element interactions for nearby clusters
    /// - Far-field: Multipole expansions with S*D*T factorization
    ///
    /// # Arguments
    /// * `x` - Input vector (length = num_dofs)
    ///
    /// # Returns
    /// * `y` - Output vector (length = num_dofs), y = A*x
    pub fn matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_into(x, &mut y);
        y
    }

    /// Apply the forward FMM action only on caller-owned rows.
    ///
    /// This is a local-action contract for distributed transports; it does
    /// not perform communication or imply a distributed solve. The checked
    /// wrapper validates dimensions before delegating to the legacy matvec.
    pub fn apply_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, SlfmmRowActionError> {
        validate_row_action_input(input, self.num_dofs, "forward")?;
        self.validate_action_layout("forward")?;
        validate_rows(rows, self.num_dofs)?;
        let output = self.matvec(input);
        validate_row_action_output(&output, "forward")?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the transpose FMM action only on caller-owned rows.
    pub fn apply_transpose_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, SlfmmRowActionError> {
        validate_row_action_input(input, self.num_dofs, "transpose")?;
        self.validate_action_layout("transpose")?;
        validate_rows(rows, self.num_dofs)?;
        let output = self.matvec_transpose(input);
        validate_row_action_output(&output, "transpose")?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the forward FMM action to selected rows for several RHS columns.
    ///
    /// This is a local ownership/action contract: it performs no transport or
    /// distributed solve, while preserving caller-supplied row order.
    pub fn apply_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, SlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "forward block")?;
        self.validate_action_layout("forward block")?;
        validate_rows(rows, self.num_dofs)?;
        let mut full_action = DenseMatrix::zeros(self.num_dofs, input.ncols());
        self.matvec_block_into(input, &mut full_action);
        validate_row_action_block_output(&full_action, "forward block")?;
        let mut output = DenseMatrix::zeros(rows.len(), input.ncols());
        for (selected, &row) in rows.iter().enumerate() {
            for rhs in 0..input.ncols() {
                output[[selected, rhs]] = full_action[[row, rhs]];
            }
        }
        Ok(output)
    }

    /// Apply the transpose FMM action to selected rows for several RHS
    /// columns, preserving caller-supplied row order.
    pub fn apply_transpose_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, SlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "transpose block")?;
        self.validate_action_layout("transpose block")?;
        validate_rows(rows, self.num_dofs)?;
        let full_action = self.matvec_transpose_block(input);
        validate_row_action_block_output(&full_action, "transpose block")?;
        let mut output = DenseMatrix::zeros(rows.len(), input.ncols());
        for (selected, &row) in rows.iter().enumerate() {
            for rhs in 0..input.ncols() {
                output[[selected, rhs]] = full_action[[row, rhs]];
            }
        }
        Ok(output)
    }

    /// Compute the matrix-vector product y = A*x into a pre-allocated buffer.
    /// Checked Hermitian-transpose SLFMM action for several right-hand sides.
    pub fn apply_hermitian_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, SlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "Hermitian block")?;
        self.validate_action_layout("Hermitian block")?;
        let mut output = DenseMatrix::zeros(self.num_dofs, input.ncols());
        for rhs in 0..input.ncols() {
            let column: Vec<_> = (0..self.num_dofs).map(|row| input[[row, rhs]]).collect();
            let result = self.apply_hermitian_checked(&column)?;
            for (row, value) in result.into_iter().enumerate() {
                output[[row, rhs]] = value;
            }
        }
        validate_row_action_block_output(&output, "Hermitian block")?;
        Ok(output)
    }

    /// Apply checked Hermitian-transpose SLFMM action on caller-owned rows.
    pub fn apply_hermitian_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, SlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "Hermitian block")?;
        self.validate_action_layout("Hermitian block")?;
        validate_rows(rows, self.num_dofs)?;
        let full_action = self.apply_hermitian_block_checked(input)?;
        let mut output = DenseMatrix::zeros(rows.len(), input.ncols());
        for (selected, &row) in rows.iter().enumerate() {
            for rhs in 0..input.ncols() {
                output[[selected, rhs]] = full_action[[row, rhs]];
            }
        }
        Ok(output)
    }

    /// Compute matrix-vector product `y = A*x` into a pre-allocated buffer.
    pub fn matvec_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.num_dofs);
        y.fill(Complex64::new(0.0, 0.0));

        // === Near-field contribution: y += [N] * x (gated parallel) ===
        let near_contributions: Vec<Vec<(usize, Complex64)>> = gated_flat_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

                let mut contributions = Vec::new();
                let x_local: Vec<Complex64> = fld_dofs.iter().map(|&i| x[i]).collect();
                let y_local = block.coefficients.matvec(&x_local);
                for (local_i, &global_i) in src_dofs.iter().enumerate() {
                    contributions.push((global_i, y_local[local_i]));
                }
                vec![contributions]
            },
        );

        for contributions in near_contributions {
            for (idx, val) in contributions {
                y[idx] += val;
            }
        }

        // === Far-field contribution: y += [S][D][T] * x (parallelized) ===
        let multipoles: Vec<Vec<Complex64>> = gated_enumerate_map(
            &self.t_matrices,
            self.t_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, t_mat| {
                let cluster_dofs = &self.cluster_dof_indices[cluster_idx];
                if cluster_dofs.is_empty() || t_mat.as_slice().is_empty() {
                    vec![Complex64::new(0.0, 0.0); self.num_sphere_points]
                } else {
                    let x_local: Vec<Complex64> = cluster_dofs.iter().map(|&i| x[i]).collect();
                    t_mat.matvec(&x_local)
                }
            },
        );

        let d_contributions: Vec<(usize, Vec<Complex64>)> = gated_map(
            &self.d_matrices,
            self.d_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |d_entry| {
                let src_mult = &multipoles[d_entry.source_cluster];
                let translated: Vec<Complex64> = d_entry
                    .diagonal
                    .iter()
                    .zip(src_mult.iter())
                    .map(|(a, b)| *a * *b)
                    .collect();
                (d_entry.field_cluster, translated)
            },
        );

        let mut locals: Vec<Vec<Complex64>> = (0..self.num_clusters)
            .map(|_| vec![Complex64::new(0.0, 0.0); self.num_sphere_points])
            .collect();

        for (field_cluster, translated) in d_contributions {
            for i in 0..self.num_sphere_points {
                locals[field_cluster][i] += translated[i];
            }
        }

        let far_contributions: Vec<Vec<(usize, Complex64)>> = gated_enumerate_filter_map(
            &self.s_matrices,
            self.s_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, s_mat| {
                let cluster_dofs = &self.cluster_dof_indices[cluster_idx];
                if cluster_dofs.is_empty() || s_mat.as_slice().is_empty() {
                    return None;
                }
                let y_local = s_mat.matvec(&locals[cluster_idx]);
                let contributions: Vec<(usize, Complex64)> = cluster_dofs
                    .iter()
                    .enumerate()
                    .map(|(local_j, &global_j)| (global_j, y_local[local_j]))
                    .collect();
                Some(contributions)
            },
        );

        for contributions in far_contributions {
            for (idx, val) in contributions {
                y[idx] += val;
            }
        }
    }

    /// Compute `Y = A · X` for multiple right-hand sides while sharing the
    /// SLFMM multipole, diagonal translation, and local-expansion stages.
    pub fn matvec_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_block_into(x, &mut y);
        y
    }

    /// Compute a multi-RHS SLFMM product into a pre-allocated row-major
    /// matrix. Near-field blocks and all three far-field stages use dense
    /// matrix-matrix products rather than calling the scalar matvec once per
    /// right-hand side.
    pub fn matvec_block_into(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        assert_eq!(x.nrows(), self.num_dofs);
        assert_eq!(y.nrows(), self.num_dofs);
        assert_eq!(y.ncols(), x.ncols());
        y.fill(Complex64::new(0.0, 0.0));
        let nrhs = x.ncols();

        let near_contributions: Vec<Vec<(usize, Vec<Complex64>)>> = gated_flat_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];
                let x_local =
                    DenseMatrix::from_fn(fld_dofs.len(), nrhs, |j, rhs| x[[fld_dofs[j], rhs]]);
                let mut y_local = DenseMatrix::<Complex64>::zeros(src_dofs.len(), nrhs);
                block.coefficients.matmul_into(&x_local, &mut y_local);
                let contributions = src_dofs
                    .iter()
                    .enumerate()
                    .map(|(local_i, &global_i)| (global_i, y_local.row(local_i).to_vec()))
                    .collect();
                vec![contributions]
            },
        );
        for contributions in near_contributions {
            for (idx, values) in contributions {
                for (rhs, value) in values.into_iter().enumerate() {
                    y[[idx, rhs]] += value;
                }
            }
        }

        let multipoles: Vec<DenseMatrix<Complex64>> = gated_enumerate_map(
            &self.t_matrices,
            self.t_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, t_mat| {
                let cluster_dofs = &self.cluster_dof_indices[cluster_idx];
                if cluster_dofs.is_empty() || t_mat.as_slice().is_empty() {
                    DenseMatrix::zeros(self.num_sphere_points, nrhs)
                } else {
                    let x_local = DenseMatrix::from_fn(cluster_dofs.len(), nrhs, |i, rhs| {
                        x[[cluster_dofs[i], rhs]]
                    });
                    let mut out = DenseMatrix::<Complex64>::zeros(self.num_sphere_points, nrhs);
                    t_mat.matmul_into(&x_local, &mut out);
                    out
                }
            },
        );

        let d_contributions: Vec<(usize, DenseMatrix<Complex64>)> = gated_map(
            &self.d_matrices,
            self.d_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |d_entry| {
                let src_mult = &multipoles[d_entry.source_cluster];
                let mut translated = DenseMatrix::<Complex64>::zeros(self.num_sphere_points, nrhs);
                for i in 0..self.num_sphere_points {
                    for rhs in 0..nrhs {
                        translated[[i, rhs]] = d_entry.diagonal[i] * src_mult[[i, rhs]];
                    }
                }
                (d_entry.field_cluster, translated)
            },
        );

        let mut locals: Vec<DenseMatrix<Complex64>> = (0..self.num_clusters)
            .map(|_| DenseMatrix::zeros(self.num_sphere_points, nrhs))
            .collect();
        for (field_cluster, translated) in d_contributions {
            for i in 0..self.num_sphere_points {
                for rhs in 0..nrhs {
                    locals[field_cluster][[i, rhs]] += translated[[i, rhs]];
                }
            }
        }

        let far_contributions: Vec<Vec<(usize, Vec<Complex64>)>> = gated_enumerate_filter_map(
            &self.s_matrices,
            self.s_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, s_mat| {
                let cluster_dofs = &self.cluster_dof_indices[cluster_idx];
                if cluster_dofs.is_empty() || s_mat.as_slice().is_empty() {
                    return None;
                }
                let mut y_local = DenseMatrix::<Complex64>::zeros(cluster_dofs.len(), nrhs);
                s_mat.matmul_into(&locals[cluster_idx], &mut y_local);
                let contributions = cluster_dofs
                    .iter()
                    .enumerate()
                    .map(|(local_i, &global_i)| (global_i, y_local.row(local_i).to_vec()))
                    .collect();
                Some(contributions)
            },
        );
        for contributions in far_contributions {
            for (idx, values) in contributions {
                for (rhs, value) in values.into_iter().enumerate() {
                    y[[idx, rhs]] += value;
                }
            }
        }
    }

    /// Apply the SLFMM operator transpose: y = A^T * x (parallelized)
    ///
    /// Used by some iterative solvers (e.g., BiCGSTAB).
    /// Compute algebraic-transpose products for several RHS columns in one
    /// SLFMM near/far traversal.
    pub fn matvec_transpose_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_transpose_block_into(x, &mut y);
        y
    }

    /// Compute `Y = Aᵀ · X` into a pre-allocated row-major matrix.
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
        if nrhs == 0 {
            return;
        }

        let near_contributions: Vec<Vec<(usize, Vec<Complex64>)>> = gated_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                let source_dofs = &self.cluster_dof_indices[block.source_cluster];
                let field_dofs = &self.cluster_dof_indices[block.field_cluster];
                let x_local = DenseMatrix::from_fn(source_dofs.len(), nrhs, |row, rhs| {
                    x[[source_dofs[row], rhs]]
                });
                let mut y_local = DenseMatrix::<Complex64>::zeros(field_dofs.len(), nrhs);
                block
                    .coefficients
                    .transposed()
                    .matmul_into(&x_local, &mut y_local);
                field_dofs
                    .iter()
                    .enumerate()
                    .map(|(row, &global)| (global, y_local.row(row).to_vec()))
                    .collect()
            },
        );
        for block_contributions in near_contributions {
            for (global, values) in block_contributions {
                for (rhs, value) in values.into_iter().enumerate() {
                    y[[global, rhs]] += value;
                }
            }
        }

        let mut locals = (0..self.num_clusters)
            .map(|_| DenseMatrix::<Complex64>::zeros(self.num_sphere_points, nrhs))
            .collect::<Vec<_>>();
        for (cluster, local) in locals.iter_mut().enumerate() {
            let Some(field_dofs) = self.cluster_dof_indices.get(cluster) else {
                continue;
            };
            let Some(s_matrix) = self.s_matrices.get(cluster) else {
                continue;
            };
            if field_dofs.is_empty() || s_matrix.as_slice().is_empty() {
                continue;
            }
            let x_local =
                DenseMatrix::from_fn(field_dofs.len(), nrhs, |row, rhs| x[[field_dofs[row], rhs]]);
            s_matrix.transposed().matmul_into(&x_local, local);
        }

        let mut multipoles = (0..self.num_clusters)
            .map(|_| DenseMatrix::<Complex64>::zeros(self.num_sphere_points, nrhs))
            .collect::<Vec<_>>();
        for entry in &self.d_matrices {
            let Some(field) = locals.get(entry.field_cluster) else {
                continue;
            };
            let Some(source) = multipoles.get_mut(entry.source_cluster) else {
                continue;
            };
            let points = self
                .num_sphere_points
                .min(entry.diagonal.len())
                .min(source.nrows())
                .min(field.nrows());
            for point in 0..points {
                for rhs in 0..nrhs {
                    source[[point, rhs]] += entry.diagonal[point] * field[[point, rhs]];
                }
            }
        }

        for (cluster, multipole) in multipoles.iter().enumerate() {
            let Some(source_dofs) = self.cluster_dof_indices.get(cluster) else {
                continue;
            };
            let Some(t_matrix) = self.t_matrices.get(cluster) else {
                continue;
            };
            if source_dofs.is_empty() || t_matrix.as_slice().is_empty() {
                continue;
            }
            let mut y_local = DenseMatrix::<Complex64>::zeros(source_dofs.len(), nrhs);
            t_matrix.transposed().matmul_into(multipole, &mut y_local);
            for (row, &global) in source_dofs.iter().enumerate() {
                for rhs in 0..nrhs {
                    y[[global, rhs]] += y_local[[row, rhs]];
                }
            }
        }
    }

    /// Compute the algebraic-transpose product `y = A^T*x`.
    pub fn matvec_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_transpose_into(x, &mut y);
        y
    }

    /// Compute the algebraic transpose product y = A^T*x into a pre-allocated buffer.
    pub fn matvec_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.num_dofs);
        y.fill(Complex64::new(0.0, 0.0));

        // === Near-field contribution (transposed, gated parallel) ===
        let near_contributions: Vec<Vec<(usize, Complex64)>> = gated_flat_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                let src_dofs = &self.cluster_dof_indices[block.source_cluster];
                let fld_dofs = &self.cluster_dof_indices[block.field_cluster];

                let mut contributions = Vec::new();
                let x_local: Vec<Complex64> = src_dofs.iter().map(|&i| x[i]).collect();
                let mut y_local = vec![Complex64::new(0.0, 0.0); fld_dofs.len()];
                block
                    .coefficients
                    .matvec_transpose_into(&x_local, &mut y_local);
                for (local_j, &global_j) in fld_dofs.iter().enumerate() {
                    contributions.push((global_j, y_local[local_j]));
                }
                vec![contributions]
            },
        );

        for contributions in near_contributions {
            for (idx, val) in contributions {
                y[idx] += val;
            }
        }

        // === Far-field contribution (transposed, parallelized): y += T^T * D^T * S^T * x ===
        let locals: Vec<Vec<Complex64>> = gated_enumerate_map(
            &self.s_matrices,
            self.s_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, s_mat| {
                let cluster_dofs = &self.cluster_dof_indices[cluster_idx];
                if cluster_dofs.is_empty() || s_mat.as_slice().is_empty() {
                    vec![Complex64::new(0.0, 0.0); self.num_sphere_points]
                } else {
                    let x_local: Vec<Complex64> = cluster_dofs.iter().map(|&i| x[i]).collect();
                    let mut out = vec![Complex64::new(0.0, 0.0); s_mat.ncols()];
                    s_mat.matvec_transpose_into(&x_local, &mut out);
                    out
                }
            },
        );

        let d_contributions: Vec<(usize, Vec<Complex64>)> = gated_map(
            &self.d_matrices,
            self.d_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |d_entry| {
                let fld_local = &locals[d_entry.field_cluster];
                let translated: Vec<Complex64> = d_entry
                    .diagonal
                    .iter()
                    .zip(fld_local.iter())
                    .map(|(a, b)| *a * *b)
                    .collect();
                (d_entry.source_cluster, translated)
            },
        );

        let mut multipoles: Vec<Vec<Complex64>> = (0..self.num_clusters)
            .map(|_| vec![Complex64::new(0.0, 0.0); self.num_sphere_points])
            .collect();

        for (source_cluster, translated) in d_contributions {
            for i in 0..self.num_sphere_points {
                multipoles[source_cluster][i] += translated[i];
            }
        }

        let far_contributions: Vec<Vec<(usize, Complex64)>> = gated_enumerate_filter_map(
            &self.t_matrices,
            self.t_matrices.len(),
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, t_mat| {
                let cluster_dofs = &self.cluster_dof_indices[cluster_idx];
                if cluster_dofs.is_empty() || t_mat.as_slice().is_empty() {
                    return None;
                }
                let mut y_local = vec![Complex64::new(0.0, 0.0); t_mat.ncols()];
                t_mat.matvec_transpose_into(&multipoles[cluster_idx], &mut y_local);
                let contributions: Vec<(usize, Complex64)> = cluster_dofs
                    .iter()
                    .enumerate()
                    .map(|(local_i, &global_i)| (global_i, y_local[local_i]))
                    .collect();
                Some(contributions)
            },
        );

        for contributions in far_contributions {
            for (idx, val) in contributions {
                y[idx] += val;
            }
        }
    }
}

fn validate_row_action_input(
    input: &[Complex64],
    expected: usize,
    operation: &'static str,
) -> Result<(), SlfmmRowActionError> {
    if input.len() != expected {
        return Err(SlfmmRowActionError::Dimension {
            operation,
            actual: input.len(),
            expected,
        });
    }
    if let Some(index) = input
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SlfmmRowActionError::NonFiniteInput { operation, index });
    }
    Ok(())
}

fn validate_row_action_output(
    output: &[Complex64],
    operation: &'static str,
) -> Result<(), SlfmmRowActionError> {
    if let Some(index) = output
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SlfmmRowActionError::NonFiniteOutput { operation, index });
    }
    Ok(())
}

fn validate_row_action_block_output(
    output: &DenseMatrix<Complex64>,
    operation: &'static str,
) -> Result<(), SlfmmRowActionError> {
    if let Some(index) = output
        .as_slice()
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SlfmmRowActionError::NonFiniteOutput { operation, index });
    }
    Ok(())
}

fn validate_row_action_block_input(
    input: &DenseMatrix<Complex64>,
    expected: usize,
    operation: &'static str,
) -> Result<(), SlfmmRowActionError> {
    if input.nrows() != expected {
        return Err(SlfmmRowActionError::Dimension {
            operation,
            actual: input.nrows(),
            expected,
        });
    }
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            let value = input[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(SlfmmRowActionError::NonFiniteInput {
                    operation,
                    index: column * input.nrows() + row,
                });
            }
        }
    }
    Ok(())
}

fn validate_rows(rows: &[usize], num_dofs: usize) -> Result<(), SlfmmRowActionError> {
    if let Some(&row) = rows.iter().find(|&&row| row >= num_dofs) {
        return Err(SlfmmRowActionError::InvalidRow { row, num_dofs });
    }
    Ok(())
}

impl LinearOperator<Complex64> for SlfmmSystem {
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

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

#[cfg(test)]
mod row_action_contract_tests {
    use super::*;

    #[test]
    fn checked_row_actions_validate_inputs_and_project_rows() {
        let system = SlfmmSystem::new(2, 0, 0, 0);
        let input = [Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.5)];

        assert_eq!(
            system.apply_rows_checked(&input, &[1, 0]).unwrap(),
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        );
        assert_eq!(
            system.apply_transpose_rows_checked(&input, &[0]).unwrap(),
            vec![Complex64::new(0.0, 0.0)]
        );
        assert!(matches!(
            system.apply_rows_checked(&input[..1], &[0]),
            Err(SlfmmRowActionError::Dimension { .. })
        ));
        assert!(matches!(
            system.apply_rows_checked(
                &[Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)],
                &[0]
            ),
            Err(SlfmmRowActionError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            system.apply_transpose_rows_checked(&input, &[2]),
            Err(SlfmmRowActionError::InvalidRow { row: 2, .. })
        ));
    }

    #[test]
    fn checked_block_row_actions_match_column_actions() {
        let system = SlfmmSystem::new(2, 0, 0, 0);
        let mut input = DenseMatrix::zeros(2, 2);
        input[[0, 0]] = Complex64::new(1.0, 0.2);
        input[[1, 0]] = Complex64::new(-2.0, 0.5);
        input[[0, 1]] = Complex64::new(0.4, -0.3);
        input[[1, 1]] = Complex64::new(0.8, 0.1);
        let rows = [1, 0, 1];
        let forward = system.apply_rows_block_checked(&input, &rows).unwrap();
        let transpose = system
            .apply_transpose_rows_block_checked(&input, &rows)
            .unwrap();
        assert_eq!(forward.nrows(), rows.len());
        assert_eq!(forward.ncols(), input.ncols());
        for column in 0..input.ncols() {
            let vector = (0..input.nrows())
                .map(|row| input[[row, column]])
                .collect::<Vec<_>>();
            let expected_forward = system.apply_rows_checked(&vector, &rows).unwrap();
            let expected_transpose = system.apply_transpose_rows_checked(&vector, &rows).unwrap();
            for row in 0..rows.len() {
                assert_eq!(forward[[row, column]], expected_forward[row]);
                assert_eq!(transpose[[row, column]], expected_transpose[row]);
            }
        }
        let wrong = DenseMatrix::zeros(1, 2);
        assert!(matches!(
            system.apply_rows_block_checked(&wrong, &rows),
            Err(SlfmmRowActionError::Dimension { .. })
        ));
    }

    #[test]
    fn checked_hermitian_block_actions_match_scalar_and_rows() {
        let system = SlfmmSystem::new(2, 0, 0, 0);
        let input = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(-2.0, 0.5),
                Complex64::new(0.4, -0.3),
                Complex64::new(0.8, 0.1),
            ],
        );
        let block = system.apply_hermitian_block_checked(&input).unwrap();
        let rows = [1, 0, 1];
        let row_block = system
            .apply_hermitian_rows_block_checked(&input, &rows)
            .unwrap();
        for rhs in 0..input.ncols() {
            let column: Vec<_> = (0..input.nrows()).map(|row| input[[row, rhs]]).collect();
            let expected = system.apply_hermitian_checked(&column).unwrap();
            for row in 0..input.nrows() {
                assert!((block[[row, rhs]] - expected[row]).norm() < 1e-12);
            }
            for (selected, &row) in rows.iter().enumerate() {
                assert!((row_block[[selected, rhs]] - expected[row]).norm() < 1e-12);
            }
        }
        assert!(matches!(
            system.apply_hermitian_block_checked(&DenseMatrix::zeros(1, 2)),
            Err(SlfmmRowActionError::Dimension { .. })
        ));
        let mut nonfinite = input.clone();
        nonfinite[[0, 1]].re = f64::NAN;
        assert!(matches!(
            system.apply_hermitian_block_checked(&nonfinite),
            Err(SlfmmRowActionError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn transpose_block_matches_scalar_on_near_and_far_fixture() {
        let mut system = SlfmmSystem::new(2, 2, 2, 0);
        system.cluster_dof_indices = vec![vec![0], vec![1]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 1,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(0.5, -0.2)]),
        });
        system.t_matrices = vec![
            DenseMatrix::from_row_major_vec(
                2,
                1,
                vec![Complex64::new(1.0, 0.1), Complex64::new(-0.3, 0.4)],
            ),
            DenseMatrix::from_row_major_vec(
                2,
                1,
                vec![Complex64::new(0.7, -0.2), Complex64::new(0.2, 0.6)],
            ),
        ];
        system.s_matrices = vec![
            DenseMatrix::from_row_major_vec(
                1,
                2,
                vec![Complex64::new(0.9, 0.3), Complex64::new(-0.1, 0.5)],
            ),
            DenseMatrix::from_row_major_vec(
                1,
                2,
                vec![Complex64::new(-0.4, 0.2), Complex64::new(0.8, -0.1)],
            ),
        ];
        system.d_matrices.push(DMatrixEntry {
            source_cluster: 0,
            field_cluster: 1,
            diagonal: vec![Complex64::new(0.6, -0.3), Complex64::new(-0.2, 0.7)],
        });
        let input = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(0.2, -0.4),
                Complex64::new(-0.7, 0.1),
                Complex64::new(0.5, 0.8),
                Complex64::new(-0.1, -0.2),
            ],
        );
        let block = system.matvec_transpose_block(&input);
        for rhs in 0..input.ncols() {
            let column = (0..input.nrows())
                .map(|row| input[[row, rhs]])
                .collect::<Vec<_>>();
            let scalar = system.matvec_transpose(&column);
            for row in 0..input.nrows() {
                assert!((block[[row, rhs]] - scalar[row]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn independent_reference_accuracy_covers_forward_transpose_and_hermitian() {
        let matrix = [
            [Complex64::new(1.0, 0.2), Complex64::new(-0.4, 0.7)],
            [Complex64::new(0.3, -0.5), Complex64::new(0.8, 0.1)],
        ];
        let mut system = SlfmmSystem::new(2, 1, 0, 0);
        system.cluster_dof_indices = vec![vec![0, 1]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                2,
                2,
                matrix.iter().flatten().copied().collect(),
            ),
        });
        let probes = vec![
            vec![Complex64::new(0.7, -0.2), Complex64::new(-0.1, 0.6)],
            vec![Complex64::new(-0.3, 0.4), Complex64::new(0.9, 0.1)],
        ];
        for action in [
            FmmAction::Forward,
            FmmAction::Transpose,
            FmmAction::Hermitian,
        ] {
            let references = probes
                .iter()
                .map(|probe| {
                    (0..2)
                        .map(|row| {
                            (0..2)
                                .map(|column| {
                                    let coefficient = match action {
                                        FmmAction::Forward => matrix[row][column],
                                        FmmAction::Transpose => matrix[column][row],
                                        FmmAction::Hermitian => matrix[column][row].conj(),
                                    };
                                    coefficient * probe[column]
                                })
                                .sum()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let error = system
                .relative_action_error(action, &probes, &references)
                .unwrap();
            assert!(error < 1e-14, "{action:?} relative error {error}");
        }

        assert!(matches!(
            system.relative_action_error(FmmAction::Forward, &[], &[]),
            Err(FmmAccuracyError::EmptyProbeSet)
        ));
        assert!(matches!(
            system.relative_action_error(
                FmmAction::Forward,
                &[vec![Complex64::new(1.0, 0.0)]],
                &[vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]],
            ),
            Err(FmmAccuracyError::InputLength { .. })
        ));
    }

    #[test]
    fn basis_checked_builder_expands_linear_panel_dofs() {
        use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};

        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let elements = vec![Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2],
        }];
        let mut cluster = Cluster::new(elements[0].center);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let system = build_slfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            &[cluster],
            &physics,
            BasisOrder::Linear,
            4,
            8,
            3,
            -1.0,
            1.0,
        )
        .expect("linear basis SLFMM admission");
        assert_eq!(system.num_dofs, 3);
        assert_eq!(system.cluster_dof_indices[0].len(), 3);
        assert_eq!(system.t_matrices[0].ncols(), 3);
        assert_eq!(system.s_matrices[0].nrows(), 3);
        let t_matrix = &system.t_matrices[0];
        let basis_column_delta: f64 = (0..t_matrix.nrows())
            .map(|row| (t_matrix[[row, 0]] - t_matrix[[row, 1]]).norm())
            .sum();
        assert!(
            basis_column_delta > 1e-12,
            "linear SLFMM T columns must retain distinct basis weights"
        );
    }

    #[test]
    fn analytic_wavenumber_derivative_matches_centered_linear_basis_action() {
        use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let elements = vec![Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2],
        }];
        let mut cluster = Cluster::new(elements[0].center);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let basis = BasisOrder::Linear;
        let n_theta = 4;
        let n_phi = 8;
        let n_terms = 3;
        let beta_scale = 1.0;
        let base = build_slfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            &[cluster.clone()],
            &physics,
            basis,
            n_theta,
            n_phi,
            n_terms,
            sign,
            beta_scale,
        )
        .expect("linear basis SLFMM assembly");
        let derivative = build_slfmm_wavenumber_derivative_with_basis_checked(
            &elements,
            &nodes,
            &[cluster.clone()],
            &physics,
            basis,
            n_theta,
            n_phi,
            n_terms,
            sign,
            beta_scale,
        )
        .expect("linear basis SLFMM derivative assembly");
        let x = vec![
            Complex64::new(0.7, -0.2),
            Complex64::new(-0.3, 0.4),
            Complex64::new(0.2, 0.1),
        ];
        let analytic = derivative.apply(&base, &x);
        let step = 1.0e-4;
        let make_physics = |wavenumber: f64| {
            PhysicsParams::new(
                wavenumber * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
                physics.speed_of_sound,
                physics.density,
                false,
            )
        };
        let plus = build_slfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            &[cluster.clone()],
            &make_physics(physics.wave_number + step),
            basis,
            n_theta,
            n_phi,
            n_terms,
            sign,
            beta_scale,
        )
        .expect("positive centered SLFMM assembly")
        .matvec(&x);
        let minus = build_slfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            &[cluster],
            &make_physics(physics.wave_number - step),
            basis,
            n_theta,
            n_phi,
            n_terms,
            sign,
            beta_scale,
        )
        .expect("negative centered SLFMM assembly")
        .matvec(&x);
        let finite_difference: Vec<_> = plus
            .iter()
            .zip(minus.iter())
            .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
            .collect();
        let error = analytic
            .iter()
            .zip(finite_difference.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = finite_difference
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        assert!(
            error / scale < 2.0e-3,
            "linear SLFMM derivative error={}",
            error / scale
        );
    }

    #[test]
    fn analytic_wavenumber_derivative_matches_centered_quadratic_action() {
        use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ];
        let element = |connectivity: Vec<usize>, dof_addresses: Vec<usize>| Element {
            connectivity,
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0], [0.1, 0.0, 0.995], [0.0, 0.1, 0.995]],
            center: [0.0, 0.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses,
        };
        let mut elements = vec![
            element(vec![0, 1, 2], (0..6).collect()),
            element(vec![3, 4, 5], (6..12).collect()),
        ];
        elements[0].center = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        elements[1].center = [13.0 / 3.0, 1.0 / 3.0, 0.0];
        let mut first = Cluster::new(elements[0].center);
        first.element_indices = vec![0];
        first.num_elements = 1;
        first.near_clusters = vec![1];
        first.far_clusters = vec![1];
        let mut second = Cluster::new(elements[1].center);
        second.element_indices = vec![1];
        second.num_elements = 1;
        second.near_clusters = vec![0];
        second.far_clusters = vec![0];
        let clusters = vec![first, second];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let basis = BasisOrder::Quadratic;
        let n_theta = 4;
        let n_phi = 8;
        let n_terms = 3;
        let base = build_slfmm_system_stabilized_with_basis_checked(
            &elements, &nodes, &clusters, &physics, basis, n_theta, n_phi, n_terms, sign, 0.2,
        )
        .expect("quadratic basis SLFMM assembly");
        let derivative = build_slfmm_wavenumber_derivative_with_basis_checked(
            &elements, &nodes, &clusters, &physics, basis, n_theta, n_phi, n_terms, sign, 0.2,
        )
        .expect("quadratic basis SLFMM derivative assembly");
        let x: Vec<_> = (0..12)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
            .collect();
        let analytic = derivative.apply(&base, &x);
        let step = 1.0e-4;
        let make_physics = |wavenumber: f64| {
            PhysicsParams::new(
                wavenumber * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
                physics.speed_of_sound,
                physics.density,
                false,
            )
        };
        let plus = build_slfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            &clusters,
            &make_physics(physics.wave_number + step),
            basis,
            n_theta,
            n_phi,
            n_terms,
            sign,
            0.2,
        )
        .expect("positive centered quadratic SLFMM assembly")
        .matvec(&x);
        let minus = build_slfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            &clusters,
            &make_physics(physics.wave_number - step),
            basis,
            n_theta,
            n_phi,
            n_terms,
            sign,
            0.2,
        )
        .expect("negative centered quadratic SLFMM assembly")
        .matvec(&x);
        let finite_difference: Vec<_> = plus
            .iter()
            .zip(minus.iter())
            .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
            .collect();
        let error = analytic
            .iter()
            .zip(finite_difference.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = finite_difference
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        assert!(
            error / scale < 2.0e-3,
            "quadratic SLFMM derivative error={}",
            error / scale
        );
    }

    #[test]
    fn curved_quad4_slfmm_proxies_carry_collocation_normals() {
        use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};

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
        let (proxies, _) =
            build_slfmm_basis_proxies(std::slice::from_ref(&element), &nodes, BasisOrder::Linear);
        assert_eq!(proxies.len(), 4);
        assert!((expected[0][2] - expected[1][2]).abs() > 1.0e-6);
        for (proxy, normal) in proxies.iter().zip(expected) {
            let error = (0..3)
                .map(|axis| (proxy.normal[axis] - normal[axis]).abs())
                .fold(0.0_f64, f64::max);
            assert!(error < 1.0e-12, "curved SLFMM proxy normal error={error}");
        }
    }

    #[test]
    fn basis_checked_builder_rejects_wrong_local_dof_count() {
        use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};

        let element = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };
        let mut cluster = Cluster::new(element.center);
        cluster.element_indices = vec![0];
        let error = match build_slfmm_system_stabilized_with_basis_checked(
            &[element],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[cluster],
            &PhysicsParams::new(100.0, 343.0, 1.21, false),
            BasisOrder::Linear,
            4,
            8,
            3,
            -1.0,
            1.0,
        ) {
            Ok(_) => panic!("wrong basis layout must fail before assembly"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            SlfmmAssemblyError::InvalidBasisDofCount {
                expected: 3,
                actual: 1,
                ..
            }
        ));
    }
}

/// Validate the constant-panel topology consumed by the SLFMM kernels.
///
/// This helper is public so callers that construct clusters themselves can
/// fail closed before invoking the legacy infallible builder. It does not
/// perform quadrature or allocate operator storage.
pub fn validate_slfmm_constant_panel_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
) -> Result<usize, SlfmmAssemblyError> {
    if n_theta == 0 || n_phi == 0 || n_terms == 0 {
        return Err(SlfmmAssemblyError::InvalidQuadrature {
            n_theta,
            n_phi,
            n_terms,
        });
    }

    let num_dofs = count_dofs(elements);
    let mut dof_owner = vec![None; num_dofs];
    for (element_index, element) in elements.iter().enumerate() {
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(SlfmmAssemblyError::InvalidNode {
                    element: element_index,
                    local,
                    node,
                    available: nodes.len(),
                });
            }
        }
        if element.property.is_evaluation() {
            continue;
        }
        if element.dof_addresses.len() != 1 {
            return Err(SlfmmAssemblyError::UnsupportedBasis {
                element: element_index,
                dofs: element.dof_addresses.len(),
            });
        }
        let global = element.dof_addresses[0];
        if global >= num_dofs {
            return Err(SlfmmAssemblyError::InvalidDofAddress {
                element: element_index,
                global,
                num_dofs,
            });
        }
        if let Some(first_element) = dof_owner[global] {
            return Err(SlfmmAssemblyError::DuplicateGlobalDof {
                first_element,
                second_element: element_index,
                global,
            });
        }
        dof_owner[global] = Some(element_index);
    }
    if let Some(global) = dof_owner.iter().position(Option::is_none) {
        return Err(SlfmmAssemblyError::InvalidDofAddress {
            element: elements.len(),
            global,
            num_dofs,
        });
    }

    let mut memberships = vec![None; elements.len()];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for &element_index in &cluster.element_indices {
            if element_index >= elements.len() {
                return Err(SlfmmAssemblyError::InvalidClusterElement {
                    cluster: cluster_index,
                    element: element_index,
                    available: elements.len(),
                });
            }
            if elements[element_index].property.is_evaluation() {
                continue;
            }
            if let Some(first_cluster) = memberships[element_index] {
                return Err(SlfmmAssemblyError::DuplicateElementMembership {
                    element: element_index,
                    first_cluster,
                    second_cluster: cluster_index,
                });
            }
            memberships[element_index] = Some(cluster_index);
        }
    }
    for (element_index, element) in elements.iter().enumerate() {
        if !element.property.is_evaluation() && memberships[element_index].is_none() {
            return Err(SlfmmAssemblyError::MissingElement {
                element: element_index,
            });
        }
    }

    Ok(num_dofs)
}

/// Build a checked SLFMM system for constant-panel elements.
///
/// The checked entry validates basis incidence, global DOF numbering, cluster
/// ownership, connectivity, and quadrature dimensions before any expensive
/// assembly work. Higher-order/shared layouts return
/// [`SlfmmAssemblyError::UnsupportedBasis`] instead of being misinterpreted as
/// one coefficient per panel.
pub fn build_slfmm_system_stabilized_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
    sign: f64,
    beta_scale: f64,
) -> Result<SlfmmSystem, SlfmmAssemblyError> {
    let num_dofs =
        validate_slfmm_constant_panel_layout(elements, nodes, clusters, n_theta, n_phi, n_terms)?;
    Ok(build_slfmm_system_stabilized_impl(
        elements, nodes, clusters, physics, n_theta, n_phi, n_terms, sign, beta_scale, num_dofs,
    ))
}

/// Build a basis-weighted SLFMM system for discontinuous or continuous
/// linear/quadratic panel traces.
///
/// The builder expands every panel-local basis slot into a proxy element. The
/// proxy retains the original geometry and boundary condition, stores its
/// collocation point in `center`, and encodes the basis slot in `group`. This
/// keeps the existing matrix-free cluster layout while making both the
/// near-field blocks and the source/field multipoles basis weighted.
pub fn build_slfmm_system_stabilized_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    basis: BasisOrder,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
    sign: f64,
    beta_scale: f64,
) -> Result<SlfmmSystem, SlfmmAssemblyError> {
    if basis == BasisOrder::Constant {
        return build_slfmm_system_stabilized_checked(
            elements, nodes, clusters, physics, n_theta, n_phi, n_terms, sign, beta_scale,
        );
    }
    let num_dofs =
        validate_slfmm_basis_layout(elements, nodes, clusters, basis, n_theta, n_phi, n_terms)?;
    let (proxy_elements, element_proxies) = build_slfmm_basis_proxies(elements, nodes, basis);
    let expanded_clusters = expand_slfmm_clusters(clusters, &element_proxies);
    Ok(build_slfmm_system_stabilized_impl(
        &proxy_elements,
        nodes,
        &expanded_clusters,
        physics,
        n_theta,
        n_phi,
        n_terms,
        sign,
        beta_scale,
        num_dofs,
    ))
}

/// Build the fixed-geometry wavenumber derivative for basis-weighted SLFMM.
///
/// The proxy expansion is identical to the value builder, so continuous
/// shared DOFs and local basis slots retain the same cluster topology.
#[allow(clippy::too_many_arguments)]
pub fn build_slfmm_wavenumber_derivative_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    basis: BasisOrder,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
    sign: f64,
    beta_scale: f64,
) -> Result<SlfmmWavenumberDerivative, SlfmmAssemblyError> {
    if basis == BasisOrder::Constant {
        return build_slfmm_wavenumber_derivative(
            elements, nodes, clusters, physics, n_theta, n_phi, n_terms, sign, beta_scale,
        )
        .map_err(|_error| SlfmmAssemblyError::InvalidQuadrature {
            n_theta,
            n_phi,
            n_terms,
        });
    }
    let num_dofs =
        validate_slfmm_basis_layout(elements, nodes, clusters, basis, n_theta, n_phi, n_terms)?;
    let (proxy_elements, element_proxies) = build_slfmm_basis_proxies(elements, nodes, basis);
    let expanded_clusters = expand_slfmm_clusters(clusters, &element_proxies);
    let derivative = build_slfmm_wavenumber_derivative(
        &proxy_elements,
        nodes,
        &expanded_clusters,
        physics,
        n_theta,
        n_phi,
        n_terms,
        sign,
        beta_scale,
    )
    .map_err(|_error| SlfmmAssemblyError::InvalidQuadrature {
        n_theta,
        n_phi,
        n_terms,
    })?;
    let _ = num_dofs;
    Ok(derivative)
}

fn validate_slfmm_basis_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    basis: BasisOrder,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
) -> Result<usize, SlfmmAssemblyError> {
    if n_theta == 0 || n_phi == 0 || n_terms == 0 {
        return Err(SlfmmAssemblyError::InvalidQuadrature {
            n_theta,
            n_phi,
            n_terms,
        });
    }
    let num_dofs = count_dofs(elements);
    let mut owned = vec![false; num_dofs];
    for (element_index, element) in elements.iter().enumerate() {
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(SlfmmAssemblyError::InvalidNode {
                    element: element_index,
                    local,
                    node,
                    available: nodes.len(),
                });
            }
        }
        if element.property.is_evaluation() {
            continue;
        }
        let expected = element.element_type.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(SlfmmAssemblyError::InvalidBasisDofCount {
                element: element_index,
                expected,
                actual: element.dof_addresses.len(),
            });
        }
        for &global in &element.dof_addresses {
            if global >= num_dofs {
                return Err(SlfmmAssemblyError::InvalidDofAddress {
                    element: element_index,
                    global,
                    num_dofs,
                });
            }
            owned[global] = true;
        }
    }
    if let Some(global) = owned.iter().position(|owned| !owned) {
        return Err(SlfmmAssemblyError::InvalidDofAddress {
            element: elements.len(),
            global,
            num_dofs,
        });
    }

    let mut memberships = vec![None; elements.len()];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for &element_index in &cluster.element_indices {
            if element_index >= elements.len() {
                return Err(SlfmmAssemblyError::InvalidClusterElement {
                    cluster: cluster_index,
                    element: element_index,
                    available: elements.len(),
                });
            }
            if elements[element_index].property.is_evaluation() {
                continue;
            }
            if let Some(first_cluster) = memberships[element_index] {
                return Err(SlfmmAssemblyError::DuplicateElementMembership {
                    element: element_index,
                    first_cluster,
                    second_cluster: cluster_index,
                });
            }
            memberships[element_index] = Some(cluster_index);
        }
    }
    for (element, value) in elements.iter().enumerate() {
        if !value.property.is_evaluation() && memberships[element].is_none() {
            return Err(SlfmmAssemblyError::MissingElement { element });
        }
    }
    Ok(num_dofs)
}

const SLFMM_PROXY_TAG: usize = 1usize << (usize::BITS - 1);
const SLFMM_PROXY_BASIS_SHIFT: usize = 8;

fn encode_slfmm_proxy_group(num_basis: usize, local: usize) -> usize {
    SLFMM_PROXY_TAG | ((num_basis & 0xff) << SLFMM_PROXY_BASIS_SHIFT) | (local & 0xff)
}

fn build_slfmm_basis_proxies(
    elements: &[Element],
    nodes: &[[f64; 3]],
    basis: BasisOrder,
) -> (Vec<Element>, Vec<Vec<usize>>) {
    let mut proxies = Vec::new();
    let mut element_proxies = vec![Vec::new(); elements.len()];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            element_proxies[element_index].push(proxies.len());
            proxies.push(element.clone());
            continue;
        }
        let points = element.collocation_points(basis, nodes);
        let normals = element.collocation_normals(basis, nodes);
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            let mut proxy = element.clone();
            proxy.center = points[local];
            proxy.normal = normals[local];
            proxy.dof_addresses = vec![dof];
            proxy.group = encode_slfmm_proxy_group(element.dof_addresses.len(), local);
            element_proxies[element_index].push(proxies.len());
            proxies.push(proxy);
        }
    }
    (proxies, element_proxies)
}

fn expand_slfmm_clusters(clusters: &[Cluster], element_proxies: &[Vec<usize>]) -> Vec<Cluster> {
    clusters
        .iter()
        .map(|cluster| {
            let mut expanded = cluster.clone();
            expanded.element_indices = cluster
                .element_indices
                .iter()
                .flat_map(|&element| element_proxies[element].iter().copied())
                .collect();
            expanded.num_elements = expanded.element_indices.len();
            expanded
        })
        .collect()
}

/// Build SLFMM system matrices.
///
/// # Arguments
/// * `elements` - Vector mesh elements
/// * `nodes` - Node coordinates (num_nodes × 3)
/// * `clusters` - Vector clusters
/// * `physics` - Physics parameters
///
///
///
pub fn build_slfmm_system_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
    sign: f64,
    beta_scale: f64,
) -> SlfmmSystem {
    build_slfmm_system_stabilized_impl(
        elements,
        nodes,
        clusters,
        physics,
        n_theta,
        n_phi,
        n_terms,
        sign,
        beta_scale,
        count_dofs(elements),
    )
}

fn build_slfmm_system_stabilized_impl(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
    sign: f64,
    beta_scale: f64,
    num_dofs: usize,
) -> SlfmmSystem {
    let num_clusters = clusters.len();

    // Compute unit sphere quadrature points first — the actual number of points
    // may differ from n_theta*n_phi if the Gauss-Legendre table falls back to
    // a lower order (e.g. order 24 falls back to 20).
    let (sphere_coords, sphere_weights) = unit_sphere_quadrature(n_theta, n_phi);
    let num_sphere_points = sphere_coords.len();

    let mut system = SlfmmSystem::new(num_dofs, num_clusters, num_sphere_points, n_terms);

    // Build cluster DOF mappings: for each cluster, collect the global DOF indices
    // of elements that belong to this cluster
    build_cluster_dof_mappings(&mut system, elements, clusters);

    // Build near-field matrix
    build_near_field_stabilized(
        &mut system,
        elements,
        nodes,
        clusters,
        physics,
        sign,
        beta_scale,
    );

    // Build T-matrices (field-side: element → multipole, with ∂G/∂n_y + area)
    build_t_matrices(
        &mut system,
        elements,
        nodes,
        clusters,
        physics,
        &sphere_coords,
        &sphere_weights,
    );

    // Build D-matrices (cluster-to-cluster translation)
    build_d_matrices(&mut system, clusters, physics, &sphere_coords, n_terms);

    // Build S-matrices (source-side: multipole → element, with Burton-Miller combo)
    build_s_matrices(
        &mut system,
        elements,
        clusters,
        physics,
        &sphere_coords,
        sign,
        beta_scale,
    );

    system
}

/// Build the mapping from clusters to global DOF indices
fn build_cluster_dof_mappings(
    system: &mut SlfmmSystem,
    elements: &[Element],
    clusters: &[Cluster],
) {
    for cluster in clusters {
        let mut dof_indices = Vec::new();
        for &elem_idx in &cluster.element_indices {
            let elem = &elements[elem_idx];
            if elem.property.is_evaluation() {
                continue;
            }
            // Collect all DOF addresses for this element
            dof_indices.extend(elem.dof_addresses.iter().copied());
        }
        system.cluster_dof_indices.push(dof_indices);
    }
}

/// Build the near-field matrix blocks in parallel with stability params
fn build_near_field_stabilized(
    system: &mut SlfmmSystem,
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) {
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);

    // Map each element to its coordinates once
    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    // Collect all cluster pairs that need processing
    let mut cluster_pairs: Vec<(usize, usize, bool)> = Vec::new();

    for (i, cluster_i) in clusters.iter().enumerate() {
        // Self-interaction
        cluster_pairs.push((i, i, true));

        // Interaction with near clusters
        for &j in &cluster_i.near_clusters {
            if j != i {
                cluster_pairs.push((i, j, false));
            }
        }
    }

    // Compute all near-field blocks (rayon above the machine-adaptive gate,
    // serial for few cluster pairs).
    let near_blocks: Vec<NearFieldBlock> = gated_map(
        &cluster_pairs,
        cluster_pairs.len(),
        BEM_BLOCKS_PER_THREAD,
        |&(i, j, is_self)| {
            let cluster_i = &clusters[i];
            let cluster_j = &clusters[j];

            let basis_weighted = cluster_i
                .element_indices
                .iter()
                .chain(cluster_j.element_indices.iter())
                .any(|&index| decode_dof_proxy_group(elements[index].group).is_some());
            let block = if basis_weighted {
                compute_near_block_stabilized_with_basis(
                    elements,
                    &all_coords,
                    &cluster_i.element_indices,
                    &cluster_j.element_indices,
                    physics,
                    gamma,
                    tau,
                    beta,
                    is_self,
                    sign,
                )
            } else {
                compute_near_block_stabilized(
                    elements,
                    &all_coords,
                    &cluster_i.element_indices,
                    &cluster_j.element_indices,
                    physics,
                    gamma,
                    tau,
                    beta,
                    is_self,
                    sign,
                )
            };

            NearFieldBlock {
                source_cluster: i,
                field_cluster: j,
                coefficients: block,
            }
        },
    );

    system.near_matrix = near_blocks;
}

/// Build T-matrices (field-side: element to cluster multipole expansion) - parallelized
///
/// The T-matrix encodes the field-side BEM operators for the Burton-Miller formulation.
/// For each field element j and quadrature direction p:
///
/// ```text
/// T[p,j] = w_p * area_j * (-ik s_p · n_y_j) * exp(-ik s_p · (y_j - c))
/// ```
///
/// This encodes the field-side BEM operator: the double-layer kernel normal
/// derivative ∂G/∂n_y at the field element, weighted by quadrature and element area.
/// The Burton-Miller combination coefficients live in the S-matrix (source side).
fn build_t_matrices(
    system: &mut SlfmmSystem,
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    sphere_weights: &[f64],
) {
    let k = physics.wave_number;
    let ik = Complex64::new(0.0, k);
    let num_sphere_points = sphere_coords.len();

    // Build all T-matrices (rayon above the machine-adaptive gate, serial
    // for few clusters).
    let t_matrices: Vec<DenseMatrix<Complex64>> =
        gated_map(clusters, clusters.len(), BEM_BLOCKS_PER_THREAD, |cluster| {
            let num_elem = cluster.element_indices.len();
            let mut t_matrix = DenseMatrix::<Complex64>::zeros(num_sphere_points, num_elem);

            for (j, &elem_idx) in cluster.element_indices.iter().enumerate() {
                let elem = &elements[elem_idx];
                if elem.property.is_evaluation() {
                    continue;
                }

                let element_coords = get_element_coords(elem, nodes);
                let quadrature = match elem.element_type {
                    crate::core::types::ElementType::Tri3 => triangle_quadrature(3),
                    crate::core::types::ElementType::Quad4 => quad_quadrature(4),
                };

                // For each integration direction on the unit sphere
                for (p, coord) in sphere_coords.iter().enumerate() {
                    let mut plane_wave_normal_integral = Complex64::new(0.0, 0.0);
                    for &(s, t, weight) in &quadrature {
                        let shape = elem.shape_functions(s, t);
                        let (shape_ds, shape_dt) = elem.shape_derivatives(s, t);
                        let (point, jacobian, normal) = quadrature_geometry(
                            &element_coords,
                            &shape,
                            &shape_ds,
                            &shape_dt,
                            None,
                        );
                        let diff = [
                            point[0] - cluster.center[0],
                            point[1] - cluster.center[1],
                            point[2] - cluster.center[2],
                        ];
                        let s_dot_diff =
                            coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                        let phase = k * s_dot_diff;
                        let plane_wave = sonium_fastmath::exp_i(-phase);
                        let s_dot_normal =
                            coord[0] * normal[0] + coord[1] * normal[1] + coord[2] * normal[2];
                        let basis_weight = basis_order_for_proxy(elem)
                            .and_then(|basis| {
                                basis_values(elem.element_type, basis, s, t)
                                    .get(proxy_local_index(elem))
                                    .copied()
                            })
                            .unwrap_or(1.0);
                        plane_wave_normal_integral +=
                            weight * jacobian * basis_weight * s_dot_normal * plane_wave;
                    }

                    t_matrix[[p, j]] = sphere_weights[p] * (-ik) * plane_wave_normal_integral;
                }
            }

            t_matrix
        });

    system.t_matrices = t_matrices;
}

/// Build D-matrices (cluster to cluster translation) - parallelized
///
/// The D-matrix is diagonal in the single-level FMM, so we only store
/// the diagonal entries. This reduces memory from O(P²) to O(P) per pair
/// where P is the number of sphere integration points.
fn build_d_matrices(
    system: &mut SlfmmSystem,
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    n_terms: usize,
) {
    let k = physics.wave_number;
    let num_sphere_points = system.num_sphere_points;

    // Collect all far cluster pairs
    let mut far_pairs: Vec<(usize, usize)> = Vec::new();
    for (i, cluster_i) in clusters.iter().enumerate() {
        for &j in &cluster_i.far_clusters {
            far_pairs.push((i, j));
        }
    }

    // Log memory estimate
    let num_pairs = far_pairs.len();
    let mem_per_pair = num_sphere_points * std::mem::size_of::<Complex64>();
    let total_mem_mb = (num_pairs * mem_per_pair) as f64 / (1024.0 * 1024.0);
    if total_mem_mb > 100.0 {
        eprintln!(
            "    D-matrices: {} far pairs × {} points = {:.1} MB",
            num_pairs, num_sphere_points, total_mem_mb
        );
    }

    // Compute all D-matrices in parallel (diagonal storage only).
    // Each diagonal entry uses the addition theorem for the Helmholtz Green's function:
    //   D_p = ik * sum_{n=0}^{L} (2n+1) * i^n * h_n^(1)(kr) * P_n(cos_theta_p)
    // The diagonal carries the Green's-function 1/(4π) normalization. The
    // unit-sphere quadrature weights in the T-matrix are normalized separately
    // to make the discrete sphere integral an average over directions.
    // where cos_theta_p = s_hat_p . r_hat (dot product of quadrature direction with
    // the unit vector from source cluster center to field cluster center).
    // Compute all D-matrices with a machine-adaptive gate (serial for few
    // far pairs).
    let d_matrices: Vec<DMatrixEntry> = gated_map(
        &far_pairs,
        far_pairs.len(),
        BEM_BLOCKS_PER_THREAD,
        |&(i, j)| {
            let cluster_i = &clusters[i];
            let cluster_j = &clusters[j];

            // Vector from source cluster j to field cluster i
            let diff = [
                cluster_i.center[0] - cluster_j.center[0],
                cluster_i.center[1] - cluster_j.center[1],
                cluster_i.center[2] - cluster_j.center[2],
            ];
            let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            let kr = k * r;

            // Unit vector from cluster j to cluster i
            let r_hat = [diff[0] / r, diff[1] / r, diff[2] / r];

            // Compute spherical Hankel functions h_n^(1)(kr) for n=0..n_terms
            // spherical_hankel_first_kind returns order values for n=0..order-1
            let hankel_order = (n_terms + 1).max(2);
            let h_funcs = spherical_hankel_first_kind(hankel_order, kr, 1.0);

            let green_normalization = 1.0 / (4.0 * std::f64::consts::PI);
            let i_unit = Complex64::new(0.0, 1.0);
            let ik = Complex64::new(0.0, k);

            // For each quadrature direction on the unit sphere, compute the translation coefficient
            let mut diagonal = vec![Complex64::new(0.0, 0.0); num_sphere_points];
            for (p, coord) in sphere_coords.iter().enumerate() {
                // cos(angle) between quadrature direction and cluster-to-cluster direction
                let cos_theta = coord[0] * r_hat[0] + coord[1] * r_hat[1] + coord[2] * r_hat[2];

                // Evaluate Legendre polynomials P_n(cos_theta) for n=0..n_terms
                let p_leg = legendre_polynomials(n_terms + 1, cos_theta);

                // Evaluate transfer function using addition theorem:
                // D_p = ik * sum_{n=0}^{n_terms} (2n+1) * i^n * h_n(kr) * P_n(cos_theta)
                let mut transfer = Complex64::new(0.0, 0.0);
                let mut i_power = Complex64::new(1.0, 0.0); // i^0 = 1

                for n in 0..=n_terms {
                    transfer += (2.0 * n as f64 + 1.0) * i_power * h_funcs[n] * p_leg[n];
                    i_power *= i_unit; // i^(n+1)
                }

                diagonal[p] = transfer * ik * green_normalization;
            }

            DMatrixEntry {
                source_cluster: j,
                field_cluster: i,
                diagonal,
            }
        },
    );

    system.d_matrices = d_matrices;
}

/// Build S-matrices (source-side: cluster multipole to element evaluation) - parallelized
///
/// The S-matrix encodes the source-side BEM operators with Burton-Miller combination.
/// For each source element i and quadrature direction p:
///
/// ```text
/// S[i,p] = sign * [γτ + β_scaled * (ik s_p · n_x_i)] * exp(ik s_p · (x_i - c))
/// ```
///
/// This combines the double-layer K term (γτ, no source normal dependence) with
/// the hypersingular H term (β_scaled * ik s·n_x, source normal derivative).
fn build_s_matrices(
    system: &mut SlfmmSystem,
    elements: &[Element],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    sign: f64,
    beta_scale: f64,
) {
    let k = physics.wave_number;
    let ik = Complex64::new(0.0, k);
    let num_sphere_points = sphere_coords.len();
    let gamma_tau = Complex64::new(physics.gamma() * physics.tau * sign, 0.0);
    let beta = physics.burton_miller_beta() * beta_scale * sign;

    // Build all S-matrices (rayon above the machine-adaptive gate, serial
    // for few clusters).
    let s_matrices: Vec<DenseMatrix<Complex64>> =
        gated_map(clusters, clusters.len(), BEM_BLOCKS_PER_THREAD, |cluster| {
            let num_elem = cluster.element_indices.len();
            let mut s_matrix = DenseMatrix::<Complex64>::zeros(num_elem, num_sphere_points);

            for (j, &elem_idx) in cluster.element_indices.iter().enumerate() {
                let elem = &elements[elem_idx];
                if elem.property.is_evaluation() {
                    continue;
                }

                // Precompute difference vector (source element center to cluster center)
                let diff: [f64; 3] = [
                    elem.center[0] - cluster.center[0],
                    elem.center[1] - cluster.center[1],
                    elem.center[2] - cluster.center[2],
                ];

                let n_i = &elem.normal;

                // For each integration direction on the unit sphere
                for (p, coord) in sphere_coords.iter().enumerate() {
                    let s_dot_diff = coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                    let exp_factor = sonium_fastmath::exp_i(k * s_dot_diff);

                    // Burton-Miller combination: γτ (K term) + β * (ik s·n_x) (H term)
                    let s_dot_ni = coord[0] * n_i[0] + coord[1] * n_i[1] + coord[2] * n_i[2];
                    let bm_factor = gamma_tau + beta * (ik * s_dot_ni);

                    s_matrix[[j, p]] = bm_factor * exp_factor;
                }
            }

            s_matrix
        });

    system.s_matrices = s_matrices;
}

/// Build the analytic fixed-geometry wavenumber derivative of a constant-panel
/// SLFMM factorization.  The cluster topology and quadrature rule are held
/// fixed, matching the value builder; no dense derivative matrix is formed.
#[allow(clippy::too_many_arguments)]
pub fn build_slfmm_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    n_theta: usize,
    n_phi: usize,
    n_terms: usize,
    sign: f64,
    beta_scale: f64,
) -> Result<SlfmmWavenumberDerivative, String> {
    if n_theta == 0 || n_phi == 0 || n_terms == 0 {
        return Err("SLFMM derivative quadrature dimensions must be non-zero".to_string());
    }
    let (sphere_coords, sphere_weights) = unit_sphere_quadrature(n_theta, n_phi);
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta() * beta_scale * sign;
    let beta_derivative = physics.burton_miller_beta_wavenumber_derivative() * beta_scale * sign;

    let all_coords: Vec<_> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();
    let mut near_matrix = Vec::new();
    for (field_cluster, cluster) in clusters.iter().enumerate() {
        let mut pairs = cluster.near_clusters.clone();
        if !pairs.contains(&field_cluster) {
            pairs.push(field_cluster);
        }
        for source_cluster in pairs {
            let Some(source) = clusters.get(source_cluster) else {
                return Err(format!(
                    "SLFMM derivative cluster {source_cluster} is out of range"
                ));
            };
            let mut coefficients = DenseMatrix::<Complex64>::zeros(
                cluster.element_indices.len(),
                source.element_indices.len(),
            );
            for (row, &source_index) in cluster.element_indices.iter().enumerate() {
                let source_element = elements.get(source_index).ok_or_else(|| {
                    format!("SLFMM derivative source element {source_index} is out of range")
                })?;
                if source_element.property.is_evaluation() {
                    continue;
                }
                for (col, &field_index) in source.element_indices.iter().enumerate() {
                    let field_element = elements.get(field_index).ok_or_else(|| {
                        format!("SLFMM derivative field element {field_index} is out of range")
                    })?;
                    if field_element.property.is_evaluation() {
                        continue;
                    }
                    coefficients[[row, col]] = compute_matrix_entry_wavenumber_derivative_curved(
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
                    .map_err(|error| error.to_string())?;
                }
            }
            near_matrix.push(NearFieldBlock {
                source_cluster: field_cluster,
                field_cluster: source_cluster,
                coefficients,
            });
        }
    }

    let k = physics.wave_number;
    let t_matrices = clusters
        .iter()
        .map(|cluster| {
            let mut matrix =
                DenseMatrix::<Complex64>::zeros(sphere_coords.len(), cluster.element_indices.len());
            for (column, &element_index) in cluster.element_indices.iter().enumerate() {
                let element = &elements[element_index];
                if element.property.is_evaluation() {
                    continue;
                }
                let element_coords = &all_coords[element_index];
                let quadrature = match element.element_type {
                    crate::core::types::ElementType::Tri3 => triangle_quadrature(3),
                    crate::core::types::ElementType::Quad4 => quad_quadrature(4),
                };
                for (point_index, direction) in sphere_coords.iter().enumerate() {
                    let mut integral = Complex64::new(0.0, 0.0);
                    let mut derivative_integral = Complex64::new(0.0, 0.0);
                    for &(s, t, weight) in &quadrature {
                        let shape = element.shape_functions(s, t);
                        let (shape_ds, shape_dt) = element.shape_derivatives(s, t);
                        let (point, jacobian, normal) =
                            quadrature_geometry(element_coords, &shape, &shape_ds, &shape_dt, None);
                        let diff = [
                            point[0] - cluster.center[0],
                            point[1] - cluster.center[1],
                            point[2] - cluster.center[2],
                        ];
                        let distance = direction[0] * diff[0]
                            + direction[1] * diff[1]
                            + direction[2] * diff[2];
                        let phase = k * distance;
                        let exponential = sonium_fastmath::exp_i(-phase);
                        let normal_projection = direction[0] * normal[0]
                            + direction[1] * normal[1]
                            + direction[2] * normal[2];
                        let basis_weight = basis_order_for_proxy(element)
                            .and_then(|basis| {
                                basis_values(element.element_type, basis, s, t)
                                    .get(proxy_local_index(element))
                                    .copied()
                            })
                            .unwrap_or(1.0);
                        let weighted = weight * jacobian * basis_weight * normal_projection;
                        integral += weighted * exponential;
                        derivative_integral +=
                            weighted * Complex64::new(0.0, -distance) * exponential;
                    }
                    let minus_ik = Complex64::new(0.0, -k);
                    let minus_i = Complex64::new(0.0, -1.0);
                    matrix[[point_index, column]] = sphere_weights[point_index]
                        * (minus_i * integral + minus_ik * derivative_integral);
                }
            }
            matrix
        })
        .collect();

    let mut d_matrices = Vec::new();
    let green_normalization = 1.0 / (4.0 * std::f64::consts::PI);
    let i_unit = Complex64::new(0.0, 1.0);
    for (field_cluster, cluster) in clusters.iter().enumerate() {
        for &source_cluster in &cluster.far_clusters {
            let source = clusters.get(source_cluster).ok_or_else(|| {
                format!("SLFMM derivative source cluster {source_cluster} is out of range")
            })?;
            let diff = [
                cluster.center[0] - source.center[0],
                cluster.center[1] - source.center[1],
                cluster.center[2] - source.center[2],
            ];
            let radius = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            if radius <= 0.0 {
                return Err("SLFMM derivative far-cluster distance must be positive".to_string());
            }
            let unit = [diff[0] / radius, diff[1] / radius, diff[2] / radius];
            let argument = k * radius;
            let hankel = spherical_hankel_first_kind((n_terms + 1).max(2), argument, 1.0);
            let hankel_derivative = spherical_hankel_derivative(&hankel, argument);
            let mut diagonal = vec![Complex64::new(0.0, 0.0); sphere_coords.len()];
            for (point_index, direction) in sphere_coords.iter().enumerate() {
                let cosine =
                    direction[0] * unit[0] + direction[1] * unit[1] + direction[2] * unit[2];
                let legendre = legendre_polynomials(n_terms + 1, cosine);
                let mut sum = Complex64::new(0.0, 0.0);
                let mut derivative_sum = Complex64::new(0.0, 0.0);
                let mut i_power = Complex64::new(1.0, 0.0);
                for order in 0..=n_terms {
                    let coefficient = Complex64::new((2 * order + 1) as f64, 0.0) * i_power;
                    sum += coefficient * hankel[order] * legendre[order];
                    derivative_sum += coefficient * hankel_derivative[order] * legendre[order];
                    i_power *= i_unit;
                }
                let prefactor = i_unit * k * green_normalization;
                let derivative_prefactor = i_unit * green_normalization;
                diagonal[point_index] =
                    derivative_prefactor * sum + prefactor * radius * derivative_sum;
            }
            d_matrices.push(DMatrixEntry {
                source_cluster,
                field_cluster,
                diagonal,
            });
        }
    }

    let gamma_tau = Complex64::new(physics.gamma() * physics.tau * sign, 0.0);
    let ik = Complex64::new(0.0, k);
    let derivative_ik = Complex64::new(0.0, 1.0);
    let s_matrices = clusters
        .iter()
        .map(|cluster| {
            let mut matrix =
                DenseMatrix::<Complex64>::zeros(cluster.element_indices.len(), sphere_coords.len());
            for (row, &element_index) in cluster.element_indices.iter().enumerate() {
                let element = &elements[element_index];
                if element.property.is_evaluation() {
                    continue;
                }
                let diff = [
                    element.center[0] - cluster.center[0],
                    element.center[1] - cluster.center[1],
                    element.center[2] - cluster.center[2],
                ];
                for (point_index, direction) in sphere_coords.iter().enumerate() {
                    let distance =
                        direction[0] * diff[0] + direction[1] * diff[1] + direction[2] * diff[2];
                    let exponential = sonium_fastmath::exp_i(k * distance);
                    let normal_projection = direction[0] * element.normal[0]
                        + direction[1] * element.normal[1]
                        + direction[2] * element.normal[2];
                    let factor = gamma_tau + beta * ik * normal_projection;
                    let derivative_factor = beta_derivative * ik * normal_projection
                        + beta * derivative_ik * normal_projection;
                    matrix[[row, point_index]] =
                        (derivative_factor + factor * derivative_ik * distance) * exponential;
                }
            }
            matrix
        })
        .collect();

    Ok(SlfmmWavenumberDerivative {
        near_matrix,
        t_matrices,
        d_matrices,
        s_matrices,
    })
}

fn spherical_hankel_derivative(values: &[Complex64], argument: f64) -> Vec<Complex64> {
    values
        .iter()
        .enumerate()
        .map(|(order, &value)| {
            if order == 0 {
                -values[1]
            } else {
                values[order - 1] - value * ((order + 1) as f64 / argument)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};

    fn make_test_cluster() -> Cluster {
        let mut cluster = Cluster::new([0.5, 0.5, 0.0]);
        cluster.element_indices = vec![0, 1];
        cluster.near_clusters = vec![];
        cluster.far_clusters = vec![];
        cluster
    }

    fn make_test_elements() -> (Vec<Element>, Vec<[f64; 3]>) {
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];

        let elem0 = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };

        let elem1 = Element {
            connectivity: vec![1, 3, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0, 2.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![1],
        };

        (vec![elem0, elem1], nodes)
    }

    #[test]
    fn analytic_wavenumber_derivative_matches_centered_slfmm_action() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ];
        let mut elements = make_test_elements().0;
        elements[0].connectivity = vec![0, 1, 2];
        elements[0].center = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        elements[1].connectivity = vec![3, 4, 5];
        elements[1].center = [13.0 / 3.0, 1.0 / 3.0, 0.0];
        let mut first = Cluster::new(elements[0].center);
        first.element_indices = vec![0];
        first.far_clusters = vec![1];
        let mut second = Cluster::new(elements[1].center);
        second.element_indices = vec![1];
        second.far_clusters = vec![0];
        let clusters = vec![first, second];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let beta_scale = 0.2;
        let base = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, 4, 8, 4, sign, beta_scale,
        );
        let derivative = build_slfmm_wavenumber_derivative(
            &elements, &nodes, &clusters, &physics, 4, 8, 4, sign, beta_scale,
        )
        .expect("analytic SLFMM derivative assembly");
        let x = vec![Complex64::new(0.7, -0.2), Complex64::new(-0.3, 0.4)];
        let analytic = derivative.apply(&base, &x);
        let step = 1.0e-4;
        let make_physics = |wavenumber: f64| {
            PhysicsParams::new(
                wavenumber * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
                physics.speed_of_sound,
                physics.density,
                false,
            )
        };
        let plus_physics = make_physics(physics.wave_number + step);
        let minus_physics = make_physics(physics.wave_number - step);
        let plus = build_slfmm_system_stabilized(
            &elements,
            &nodes,
            &clusters,
            &plus_physics,
            4,
            8,
            4,
            sign,
            beta_scale,
        )
        .matvec(&x);
        let minus = build_slfmm_system_stabilized(
            &elements,
            &nodes,
            &clusters,
            &minus_physics,
            4,
            8,
            4,
            sign,
            beta_scale,
        )
        .matvec(&x);
        let finite_difference: Vec<_> = plus
            .iter()
            .zip(minus.iter())
            .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
            .collect();
        let error = analytic
            .iter()
            .zip(finite_difference.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = finite_difference
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        assert!(
            error / scale < 2.0e-3,
            "SLFMM analytic derivative relative error={}",
            error / scale
        );
    }

    #[test]
    fn checked_near_field_rejects_invalid_cluster() {
        let mut system = SlfmmSystem::new(1, 1, 1, 1);
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.0, 0.0)]),
        });

        assert!(matches!(
            system.try_extract_near_field_matrix(),
            Err(SlfmmNearFieldError::InvalidCluster { .. })
        ));
    }

    #[test]
    fn checked_near_field_rejects_invalid_block_shape() {
        let mut system = SlfmmSystem::new(2, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![0, 1]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.0, 0.0)]),
        });

        assert!(matches!(
            system.try_extract_near_field_matrix(),
            Err(SlfmmNearFieldError::BlockShape { .. })
        ));
    }

    #[test]
    fn checked_near_field_rejects_out_of_range_global_dof() {
        let mut system = SlfmmSystem::new(1, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![1]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(1.0, 0.0)]),
        });

        assert!(matches!(
            system.try_extract_near_field_matrix(),
            Err(SlfmmNearFieldError::GlobalDof { .. })
        ));
    }

    #[test]
    fn checked_near_field_rejects_non_finite_coefficient() {
        let mut system = SlfmmSystem::new(1, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![0]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(f64::NAN, 0.0)],
            ),
        });

        assert!(matches!(
            system.try_extract_near_field_matrix(),
            Err(SlfmmNearFieldError::NonFiniteCoefficient { .. })
        ));
    }

    #[test]
    fn checked_action_rejects_malformed_hierarchy_before_indexing() {
        let mut system = SlfmmSystem::new(1, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![1]];
        system.t_matrices = vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )];
        system.s_matrices = vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )];

        assert!(matches!(
            system.matvec_checked(&[Complex64::new(1.0, 0.0)]),
            Err(SlfmmRowActionError::InvalidDof { .. })
        ));
    }

    #[test]
    fn checked_action_rejects_malformed_expansion_shape() {
        let mut system = SlfmmSystem::new(1, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![0]];
        system.t_matrices = vec![DenseMatrix::zeros(1, 0)];
        system.s_matrices = vec![DenseMatrix::zeros(1, 1)];

        assert!(matches!(
            system.matvec_checked(&[Complex64::new(1.0, 0.0)]),
            Err(SlfmmRowActionError::ExpansionShape { stage: "T", .. })
        ));
    }

    #[test]
    fn checked_action_rejects_non_finite_output() {
        let mut system = SlfmmSystem::new(1, 1, 1, 1);
        system.cluster_dof_indices = vec![vec![0]];
        system.t_matrices = vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )];
        system.s_matrices = vec![DenseMatrix::from_row_major_vec(
            1,
            1,
            vec![Complex64::new(1.0, 0.0)],
        )];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(f64::NAN, 0.0)],
            ),
        });

        assert!(matches!(
            system.matvec_checked(&[Complex64::new(1.0, 0.0)]),
            Err(SlfmmRowActionError::NonFiniteOutput {
                operation: "forward",
                ..
            })
        ));
    }

    #[test]
    fn test_slfmm_system_creation() {
        let system = SlfmmSystem::new(10, 2, 32, 5);
        assert_eq!(system.num_dofs, 10);
        assert_eq!(system.num_sphere_points, 32);
        assert_eq!(system.num_expansion_terms, 5);
    }

    #[test]
    fn test_build_slfmm_system() {
        let (elements, nodes) = make_test_elements();
        let cluster = make_test_cluster();
        let clusters = vec![cluster];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let system = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, 4, 8, 5, 1.0, 1.0,
        );

        assert_eq!(system.num_dofs, 2);
        assert_eq!(system.t_matrices.len(), 1);
        assert_eq!(system.s_matrices.len(), 1);
    }

    #[test]
    fn test_near_field_free_terms_match_tbem_diagonal() {
        // For a single-cluster system (all near-field), the extracted matrix
        // diagonal must match the TBEM dense matrix diagonal.
        let (elements, nodes) = make_test_elements();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let mut cluster = make_test_cluster();
        cluster.near_clusters = vec![0]; // only self
        cluster.far_clusters = vec![];
        let clusters = vec![cluster];

        let slfmm = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, 4, 8, 3, 1.0, 1.0,
        );
        let tbem = crate::core::assembly::tbem::build_tbem_system(&elements, &nodes, &physics);

        let nf = slfmm.extract_near_field_matrix();
        for i in 0..2 {
            let diff = (nf[[i, i]] - tbem.matrix[[i, i]]).norm();
            assert!(
                diff < 1e-6,
                "SLFMM diagonal[{i},{i}]={} vs TBEM[{i},{i}]={} diff={diff}",
                nf[[i, i]],
                tbem.matrix[[i, i]],
            );
        }
    }

    #[test]
    fn test_near_field_admittance_matches_tbem() {
        // FMM near-field blocks are used as the exact local correction, so
        // admittance boundaries must match dense TBEM, including off-diagonal
        // pressure-admittance coupling terms.
        let (mut elements, nodes) = make_test_elements();
        elements[0].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.0, 0.0)],
            admittance: Complex64::new(0.2, -0.04),
        };
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let mut cluster = make_test_cluster();
        cluster.near_clusters = vec![0];
        cluster.far_clusters = vec![];
        let clusters = vec![cluster];

        let slfmm = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, 4, 8, 3, 1.0, 1.0,
        );
        let tbem = crate::core::assembly::tbem::build_tbem_system(&elements, &nodes, &physics);
        let nf = slfmm.extract_near_field_matrix();

        for i in 0..2 {
            for j in 0..2 {
                let diff = (nf[[i, j]] - tbem.matrix[[i, j]]).norm();
                let scale = tbem.matrix[[i, j]].norm().max(1e-10);
                assert!(
                    diff / scale < 1e-6,
                    "admittance near-field[{i},{j}]={} vs tbem[{i},{j}]={} rel_diff={}",
                    nf[[i, j]],
                    tbem.matrix[[i, j]],
                    diff / scale,
                );
            }
        }
    }

    #[test]
    fn test_extract_near_field_no_double_count() {
        // With two clusters that are mutual near-neighbors, the extracted
        // near-field matrix must not double-count off-diagonal entries.
        let (elements, nodes) = make_test_elements();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let mut cluster0 = Cluster::new([0.5, 0.33, 0.0]);
        cluster0.element_indices = vec![0];
        cluster0.near_clusters = vec![1]; // near to cluster 1
        cluster0.far_clusters = vec![];

        let mut cluster1 = Cluster::new([1.0, 0.67, 0.0]);
        cluster1.element_indices = vec![1];
        cluster1.near_clusters = vec![0]; // near to cluster 0
        cluster1.far_clusters = vec![];

        let clusters = vec![cluster0, cluster1];
        let slfmm = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, 4, 8, 3, 1.0, 1.0,
        );

        let tbem = crate::core::assembly::tbem::build_tbem_system(&elements, &nodes, &physics);
        let nf = slfmm.extract_near_field_matrix();

        for i in 0..2 {
            for j in 0..2 {
                let diff = (nf[[i, j]] - tbem.matrix[[i, j]]).norm();
                let scale = tbem.matrix[[i, j]].norm().max(1e-10);
                assert!(
                    diff / scale < 0.1,
                    "near-field[{i},{j}]={} vs tbem[{i},{j}]={} rel_diff={}",
                    nf[[i, j]],
                    tbem.matrix[[i, j]],
                    diff / scale,
                );
            }
        }
    }

    /// Validates that multi-cluster SLFMM (with actual far-field interactions via S*D*T)
    /// produces results consistent with TBEM dense matvec.
    #[test]
    #[ignore = "slow dense-vs-SLFMM validation; run `cargo bench -p sonium-bem --bench fmm_slow_validation`"]
    fn test_multi_cluster_far_field_vs_tbem() {
        use crate::core::mesh::generators::generate_icosphere_mesh;
        use crate::core::mesh::octree::Octree;
        use crate::core::types::BoundaryCondition;
        use std::collections::HashMap;

        let radius = 0.1;
        let frequency = 500.0;
        let speed_of_sound = 343.0;
        let density = 1.21;

        let mesh = generate_icosphere_mesh(radius, 2); // 320 elements
        let n = mesh.elements.len();

        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        // Build TBEM reference
        let tbem = crate::core::assembly::tbem::build_tbem_system_stabilized(
            &elements,
            &mesh.nodes,
            &physics,
            -1.0,
            1.0,
        );

        // Build octree with small leaf size to force multiple clusters with far interactions.
        // Separation ratio 1.8 provides good accuracy for the plane-wave approximation.
        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let mut octree = Octree::build(&centers, 16, 8);
        octree.compute_interaction_lists(1.8);

        let leaf_indices = octree.leaves();
        let mut node_to_cluster: HashMap<usize, usize> = HashMap::new();
        for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
            node_to_cluster.insert(node_idx, cluster_idx);
        }

        let mut clusters: Vec<Cluster> = Vec::new();
        let mut total_far_pairs = 0;
        for &node_idx in &leaf_indices {
            let node = &octree.nodes[node_idx];
            let mut cluster = Cluster::new(node.center.clone());
            cluster.element_indices = node.element_indices.clone();
            cluster.num_elements = node.element_indices.len();
            cluster.radius = node.radius();
            cluster.near_clusters = node
                .near_clusters
                .iter()
                .filter_map(|ni| node_to_cluster.get(ni).copied())
                .collect();
            cluster.far_clusters = node
                .far_clusters
                .iter()
                .filter_map(|ni| node_to_cluster.get(ni).copied())
                .collect();
            total_far_pairs += cluster.far_clusters.len();
            clusters.push(cluster);
        }

        assert!(
            total_far_pairs > 0,
            "Test requires at least one far cluster pair; got {} clusters with 0 far pairs",
            clusters.len()
        );

        // Build SLFMM with octree clusters
        let n_terms = 10;
        let n_theta = 2 * n_terms;
        let n_phi = 2 * n_theta;
        let slfmm = build_slfmm_system_stabilized(
            &elements,
            &mesh.nodes,
            &clusters,
            &physics,
            n_theta,
            n_phi,
            n_terms,
            -1.0,
            1.0,
        );

        // Test vector
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
            .collect();

        let y_tbem = tbem.matrix.matvec(&x);
        let y_slfmm = slfmm.matvec(&x);

        let diff_norm: f64 = y_tbem
            .iter()
            .zip(y_slfmm.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let tbem_norm: f64 = y_tbem.iter().map(|y| y.norm_sqr()).sum::<f64>().sqrt();
        let rel_error = diff_norm / tbem_norm.max(1e-15);

        assert!(
            rel_error < 0.10,
            "Multi-cluster SLFMM vs TBEM relative error {rel_error:.4e} exceeds 10% threshold \
             ({} clusters, {total_far_pairs} far pairs)",
            clusters.len()
        );
    }

    #[test]
    fn test_near_field_block() {
        let (elements, nodes) = make_test_elements();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();

        let all_coords: Vec<DenseMatrix<f64>> = elements
            .iter()
            .map(|e| crate::core::assembly::common::get_element_coords(e, &nodes))
            .collect();

        let block = crate::core::assembly::common::compute_near_block_stabilized(
            &elements,
            &all_coords,
            &[0, 1],
            &[0, 1],
            &physics,
            gamma,
            tau,
            beta,
            true,
            1.0, // sign
        );

        assert_eq!(block.shape(), (2, 2));
        // Diagonal entries should be non-zero
        assert!(block[[0, 0]].norm() > 0.0);
        assert!(block[[1, 1]].norm() > 0.0);
    }

    /// When n_theta is an unsupported Gauss-Legendre order (e.g. 24), the quadrature
    /// falls back to the nearest available order (20). The system must use the actual
    /// quadrature output size, not the requested n_theta * n_phi.
    #[test]
    fn test_generated_quadrature_preserves_requested_size() {
        use crate::core::integration::unit_sphere_quadrature;

        let (elements, nodes) = make_test_elements();
        let mut cluster = make_test_cluster();
        cluster.near_clusters = vec![0];
        cluster.far_clusters = vec![];
        let clusters = vec![cluster];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let n_theta = 24; // generated because this is not a checked-in GL table
        let n_phi = 48;

        let (sphere_coords, _) = unit_sphere_quadrature(n_theta, n_phi);

        let system = build_slfmm_system_stabilized(
            &elements, &nodes, &clusters, &physics, n_theta, n_phi, 5, 1.0, 1.0,
        );

        // The system must report the requested quadrature size exactly. A
        // hidden fallback would make adjacent-order accuracy controls
        // indistinguishable and invalidate their convergence checks.
        assert_eq!(system.num_sphere_points, n_theta * n_phi);
        assert_eq!(system.num_sphere_points, sphere_coords.len());

        // T and S matrices must have dimensions consistent with the actual quadrature
        assert_eq!(system.t_matrices[0].nrows(), sphere_coords.len());
        assert_eq!(system.s_matrices[0].ncols(), sphere_coords.len());
    }

    /// Increasing n_terms must reduce the far-field contribution error vs TBEM.
    /// We isolate the far-field path by subtracting the near-field (which is identical
    /// across all runs) from the total matvec.
    #[test]
    fn test_far_field_convergence() {
        use crate::core::mesh::generators::generate_icosphere_mesh;
        use crate::core::mesh::octree::Octree;
        use std::collections::HashMap;

        let radius = 0.1;
        let frequency = 2000.0;
        let separation_ratio = 1.8;

        let mesh = generate_icosphere_mesh(radius, 2);
        let n = mesh.elements.len();

        let mut elements = mesh.elements.clone();
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
        }

        let physics = PhysicsParams::new(frequency, 343.0, 1.21, false);

        let tbem = crate::core::assembly::tbem::build_tbem_system_stabilized(
            &elements,
            &mesh.nodes,
            &physics,
            -1.0,
            1.0,
        );

        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let mut octree = Octree::build(&centers, 16, 8);
        octree.compute_interaction_lists(separation_ratio);

        let leaf_indices = octree.leaves();
        let mut node_to_cluster: HashMap<usize, usize> = HashMap::new();
        for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
            node_to_cluster.insert(node_idx, cluster_idx);
        }

        let mut clusters: Vec<Cluster> = Vec::new();
        for &node_idx in &leaf_indices {
            let node = &octree.nodes[node_idx];
            let mut cluster = Cluster::new(node.center.clone());
            cluster.element_indices = node.element_indices.clone();
            cluster.num_elements = node.element_indices.len();
            cluster.radius = node.radius();
            cluster.near_clusters = node
                .near_clusters
                .iter()
                .filter_map(|ni| node_to_cluster.get(ni).copied())
                .collect();
            cluster.far_clusters = node
                .far_clusters
                .iter()
                .filter_map(|ni| node_to_cluster.get(ni).copied())
                .collect();
            clusters.push(cluster);
        }

        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
            .collect();
        let y_tbem = tbem.matrix.matvec(&x);

        // Compute the near-field contribution once (shared across all n_terms).
        // Use the lowest n_terms just to build cluster DOF mappings and near-field.
        let slfmm_base = build_slfmm_system_stabilized(
            &elements,
            &mesh.nodes,
            &clusters,
            &physics,
            6,
            12,
            3,
            -1.0,
            1.0,
        );
        let y_near = slfmm_base.extract_near_field_matrix().matvec(&x);

        // The "ideal" far-field contribution is what TBEM computes minus the near-field
        let y_far_ref: Vec<Complex64> = y_tbem
            .iter()
            .zip(y_near.iter())
            .map(|(a, b)| a - b)
            .collect();
        let far_ref_norm: f64 = y_far_ref.iter().map(|y| y.norm_sqr()).sum::<f64>().sqrt();

        let mut errors: Vec<f64> = Vec::new();
        let mut final_far: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
        for n_terms in [3, 6, 10] {
            let n_theta = 2 * n_terms;
            let n_phi = 2 * n_theta;
            let slfmm = build_slfmm_system_stabilized(
                &elements,
                &mesh.nodes,
                &clusters,
                &physics,
                n_theta,
                n_phi,
                n_terms,
                -1.0,
                1.0,
            );

            // Extract the far-field-only contribution from SLFMM
            let y_slfmm_total = slfmm.matvec(&x);
            let y_slfmm_near = slfmm.extract_near_field_matrix().matvec(&x);
            let y_slfmm_far: Vec<Complex64> = y_slfmm_total
                .iter()
                .zip(y_slfmm_near.iter())
                .map(|(a, b)| a - b)
                .collect();

            let diff_norm: f64 = y_far_ref
                .iter()
                .zip(y_slfmm_far.iter())
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            errors.push(diff_norm / far_ref_norm.max(1e-15));
            final_far = y_slfmm_far;
        }

        // The final error (n_terms=10) must be lower than the first (n_terms=3)
        assert!(
            errors[2] < errors[0],
            "Far-field error should decrease with more expansion terms: \
             n_terms sequence [3,6,10], errors = {errors:?}",
        );
        assert!(
            errors[2] < 0.01,
            "ten-term SLFMM far field must match dense TBEM within 1%; errors={errors:?}",
        );

        let scaled_down: Vec<Complex64> = final_far
            .iter()
            .map(|v| v / (4.0 * std::f64::consts::PI))
            .collect();
        let scaled_up: Vec<Complex64> = final_far
            .iter()
            .map(|v| v * (4.0 * std::f64::consts::PI))
            .collect();

        let scaled_down_error = y_far_ref
            .iter()
            .zip(scaled_down.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / far_ref_norm.max(1e-15);
        let scaled_up_error = y_far_ref
            .iter()
            .zip(scaled_up.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / far_ref_norm.max(1e-15);

        assert!(
            errors[2] < 0.5 * scaled_down_error.min(scaled_up_error),
            "SLFMM far-field normalization is closer to a 4π-scaled variant than expected: \
             unscaled={:.4e}, scaled_down={:.4e}, scaled_up={:.4e}",
            errors[2],
            scaled_down_error,
            scaled_up_error,
        );
    }

    /// D-matrices for opposite directions must differ, confirming that the
    /// addition theorem (which depends on cos_theta = s_hat . r_hat) is
    /// direction-sensitive. This validates the source/field label fix.
    #[test]
    fn test_d_matrix_direction_matters() {
        use crate::core::integration::unit_sphere_quadrature;

        let physics = PhysicsParams::new(500.0, 343.0, 1.21, false);
        let n_terms = 5;
        let n_theta = 10;
        let n_phi = 20;
        let (sphere_coords, _) = unit_sphere_quadrature(n_theta, n_phi);
        let num_sphere_points = sphere_coords.len();

        // Two clusters well separated along the x-axis
        let mut cluster_a = Cluster::new([-1.0, 0.0, 0.0]);
        let mut cluster_b = Cluster::new([1.0, 0.0, 0.0]);
        cluster_a.element_indices = vec![0];
        cluster_b.element_indices = vec![1];

        // Forward pair: A(field) <- B(source), i.e. cluster_a is field_cluster=0, cluster_b is source
        cluster_a.far_clusters = vec![1];
        cluster_b.far_clusters = vec![0];
        cluster_a.near_clusters = vec![];
        cluster_b.near_clusters = vec![];

        let clusters_fwd = vec![cluster_a.clone(), cluster_b.clone()];

        let (elements, _nodes) = make_test_elements();

        let mut system_fwd = SlfmmSystem::new(2, 2, num_sphere_points, n_terms);
        build_cluster_dof_mappings(&mut system_fwd, &elements, &clusters_fwd);
        build_d_matrices(
            &mut system_fwd,
            &clusters_fwd,
            &physics,
            &sphere_coords,
            n_terms,
        );

        // There should be 2 D-matrix entries (A->B and B->A)
        assert_eq!(system_fwd.d_matrices.len(), 2);

        // Find the two entries by their (source, field) pair
        let d_ab = system_fwd
            .d_matrices
            .iter()
            .find(|d| d.source_cluster == 1 && d.field_cluster == 0)
            .expect("D-matrix for B->A not found");
        let d_ba = system_fwd
            .d_matrices
            .iter()
            .find(|d| d.source_cluster == 0 && d.field_cluster == 1)
            .expect("D-matrix for A->B not found");

        // The diagonals must differ because the direction vector r_hat is flipped,
        // which changes cos_theta = s_hat . r_hat for odd Legendre polynomials.
        let diff: f64 = d_ab
            .diagonal
            .iter()
            .zip(d_ba.diagonal.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let norm: f64 = d_ab
            .diagonal
            .iter()
            .map(|d| d.norm_sqr())
            .sum::<f64>()
            .sqrt();

        assert!(
            diff / norm.max(1e-15) > 1e-6,
            "D-matrices for opposite directions should differ; relative diff = {:.2e}",
            diff / norm.max(1e-15),
        );
    }

    /// End-to-end guard: SLFMM assembled with beta_scale != 1.0 must match
    /// dense TBEM to near-field precision.  Regression for the double-scaling
    /// bug where FMM near-field blocks used beta^2 instead of beta.
    #[test]
    fn test_slfmm_beta_scale_matches_dense() {
        use crate::core::mesh::generators::generate_icosphere_mesh;
        use crate::core::mesh::octree::Octree;
        use std::collections::HashMap;

        let mesh = generate_icosphere_mesh(1.0, 1); // 80 elements
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }

        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta_scale = 2.5; // deliberately non-unity

        // Dense reference
        let beta = physics.burton_miller_beta_scaled(beta_scale);
        let dense = crate::core::assembly::tbem::build_tbem_system_with_beta(
            &elements,
            &mesh.nodes,
            &physics,
            beta,
        );

        // SLFMM with the same beta_scale
        let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(1.8);

        let leaf_indices = octree.leaves();
        let mut node_to_cluster: HashMap<usize, usize> = HashMap::new();
        for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
            node_to_cluster.insert(node_idx, cluster_idx);
        }

        let mut clusters: Vec<Cluster> = Vec::with_capacity(leaf_indices.len());
        for &node_idx in &leaf_indices {
            let node = &octree.nodes[node_idx];
            let mut cluster = Cluster::new(node.center.clone());
            cluster.element_indices = node.element_indices.clone();
            cluster.num_elements = node.element_indices.len();
            cluster.radius = node.radius();
            cluster.near_clusters = node
                .near_clusters
                .iter()
                .filter_map(|ni| node_to_cluster.get(ni).copied())
                .collect();
            cluster.far_clusters = node
                .far_clusters
                .iter()
                .filter_map(|ni| node_to_cluster.get(ni).copied())
                .collect();
            clusters.push(cluster);
        }

        let characteristic_radius = clusters.iter().map(|c| c.radius).fold(0.0f64, f64::max);
        let ka = physics.wave_number * characteristic_radius;
        let n_terms = (ka.ceil() as usize + 3).clamp(5, 20);
        let n_theta = 2 * n_terms;
        let n_phi = 2 * n_theta;

        let slfmm = build_slfmm_system_stabilized(
            &elements,
            &mesh.nodes,
            &clusters,
            &physics,
            n_theta,
            n_phi,
            n_terms,
            1.0,
            beta_scale,
        );

        // Compare matvec against a random vector — near-field must match
        // exactly (same integration), far-field should be close.
        let x: Vec<Complex64> = (0..elements.len())
            .map(|i| Complex64::new(((i + 1) as f64).sin(), ((i + 2) as f64).cos()))
            .collect();

        let y_dense = dense.matrix.matvec(&x);
        let y_slfmm = slfmm.matvec(&x);

        let diff_norm: f64 = y_dense
            .iter()
            .zip(y_slfmm.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let dense_norm: f64 = y_dense.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let rel = diff_norm / dense_norm.max(1e-15);

        // Tolerance: SLFMM far-field is approximate; 5% is generous enough
        // to pass while catching a 2x beta scaling error (which would give
        // ~50-100% error on the Burton-Miller terms).
        assert!(
            rel < 0.05,
            "SLFMM vs dense TBEM with beta_scale={beta_scale}: rel error {rel:.4e} > 5%"
        );

        let x_block = DenseMatrix::from_fn(elements.len(), 3, |i, rhs| {
            Complex64::new(
                ((i + 1 + rhs * elements.len()) as f64).sin(),
                ((i + 2 + rhs) as f64).cos(),
            )
        });
        let y_block = slfmm.matvec_block(&x_block);
        let mut y_trait = DenseMatrix::<Complex64>::zeros(elements.len(), 3);
        <SlfmmSystem as LinearOperator<Complex64>>::apply_block(&slfmm, &x_block, &mut y_trait);
        for rhs in 0..x_block.ncols() {
            let expected = slfmm.matvec(&x_block.column_to_vec(rhs));
            for i in 0..elements.len() {
                assert!((y_block[[i, rhs]] - expected[i]).norm() < 1e-10);
                assert!((y_trait[[i, rhs]] - expected[i]).norm() < 1e-10);
            }
        }
    }
}
