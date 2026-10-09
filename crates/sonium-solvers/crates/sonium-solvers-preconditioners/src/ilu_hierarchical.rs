//! Hierarchical ILU (H-ILU) preconditioner
//!
//! Parallelizes the ILU factorization setup phase using nested dissection.
//! The matrix is recursively partitioned into independent blocks; blocks at
//! each level of the elimination tree are factorized in parallel via rayon.
//!
//! # Algorithm
//!
//! 1. **Partition**: BFS-based nested dissection splits DOFs into a binary tree.
//!    Separator nodes connect child subdomains.
//! 2. **Factorize**: Leaf blocks are independent → factorized in parallel.
//!    Separator blocks at each level are also independent → parallel.
//! 3. **Apply**: Block-wise forward/backward substitution, parallel within each
//!    tree level.
//!
//! # Schur-complement coupling (ILUM-style)
//!
//! The default constructors ([`HiluPreconditioner::from_csr`],
//! [`HiluPreconditioner::from_csr_with_config`]) build a true
//! Bollhöfer–Saad ILUM-style preconditioner. Unknowns are reordered as
//! `[leaf interiors …, separators …]`, giving the 2×2 block system
//!
//! ```text
//! [ B  F ]   B block-diagonal (leaves are disconnected by construction)
//! [ E  C ]
//! ```
//!
//! Each leaf block `B_i` is factored with ILU(0) as before, then the
//! approximate Schur complement `S ≈ C − E·B⁻¹·F` is formed sparsely
//! (per-leaf triangular solves against the coupling columns, with the same
//! magnitude-based dropping discipline as ILU), factored with ILU(0), and
//! `apply` performs the exact 2×2 block forward/backward substitution:
//! solve leaves, update the separator RHS, solve the separator system,
//! back-substitute the leaves.
//!
//! The historical *uncoupled* mode — each tree node (leaves *and*
//! separators) factored independently, coupling blocks discarded — remains
//! available as [`HiluPreconditioner::from_csr_uncoupled`] /
//! [`HiluPreconditioner::from_csr_uncoupled_with_config`]. It is
//! functionally block-Jacobi-ILU(0): cheaper to build, but convergence
//! degrades as the block count grows.

use std::collections::{BTreeMap, BTreeSet};

use num_traits::{Float, FromPrimitive, One, Zero};
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sparse::partition::{
    DissectionNode, build_adjacency, extract_submatrix, nested_dissection,
};

use super::ilu::{
    Ilu0Factors, IluCheckpointReader, IluDiagnostics, checkpoint_push_real,
    checkpoint_push_u32_vec, checkpoint_push_u64, checkpoint_push_usize_vec,
    checkpoint_push_values, finish_scaled_norm, ilu_checkpoint_error, ilu0_factorize,
    update_scaled_norm, validate_checkpoint_factors,
};

/// Version of the portable hierarchical-ILU factor checkpoint format.
pub const HILU_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const HILU_CHECKPOINT_MAGIC: &[u8; 8] = b"SHILUCP1";

/// Raw pointer for parallel scatter writes into disjoint output ranges.
///
/// # Safety invariant
/// Every thread must write a disjoint index set derived from a partition
/// table (`leaf_base`, block ranges, or coloring) that the caller has
/// validated: ranges are pairwise non-overlapping and in bounds. A single
/// overlapping range becomes a data race that Miri cannot easily cover
/// under rayon scheduling. Each use site carries a `debug_assert` on the
/// partition table; prefer `par_chunks_mut` / `split_at_mut` structure for
/// new code.
#[derive(Clone, Copy)]
struct DisjointScatterPtr<T>(*mut T);

unsafe impl<T: Send> Send for DisjointScatterPtr<T> {}
unsafe impl<T: Send> Sync for DisjointScatterPtr<T> {}

impl<T> DisjointScatterPtr<T> {
    unsafe fn write(self, index: usize, value: T) {
        unsafe {
            self.0.add(index).write(value);
        }
    }
}

/// Debug-only check that `base` partitions `[0, len)`: ascending, starting
/// at 0, ending at most at `len`. Used at every `DisjointScatterPtr` site.
#[cfg(debug_assertions)]
fn debug_assert_partition(base: &[usize], len: usize) {
    debug_assert!(!base.is_empty(), "partition table must be non-empty");
    debug_assert_eq!(base[0], 0, "partition must start at 0");
    debug_assert!(
        base.windows(2).all(|w| w[0] <= w[1]),
        "partition ranges must be ascending"
    );
    debug_assert!(
        base.iter().all(|&b| b <= len),
        "partition ranges must be in bounds"
    );
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the H-ILU preconditioner.
#[derive(Debug, Clone)]
pub struct HiluConfig {
    /// Minimum block size at leaf level. Default: 256.
    pub min_block_size: usize,
    /// Maximum recursion depth (None = auto from matrix size).
    pub max_levels: Option<usize>,
}

impl Default for HiluConfig {
    fn default() -> Self {
        Self {
            min_block_size: 256,
            max_levels: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: per-block ILU factors
// ---------------------------------------------------------------------------

/// ILU(0) factors for a single block.
///
/// Thin wrapper around the canonical factorization ([`ilu0_factorize`]),
/// shared with `IluPreconditioner` and the parallel ILU variants.
struct BlockFactors<T: ComplexField> {
    factors: Ilu0Factors<T>,
}

impl<T: ComplexField> BlockFactors<T> {
    /// Factorize a local CSR submatrix using the canonical ILU(0) kernel.
    fn from_local_csr(matrix: &CsrMatrix<T>) -> Self {
        Self {
            factors: ilu0_factorize(matrix),
        }
    }

    /// Forward-backward solve in place.
    fn solve_in_place(&self, x: &mut [T]) {
        assert_eq!(
            x.len(),
            self.factors.n,
            "HILU block solve buffer length mismatch"
        );
        self.factors.solve_in_place(x);
    }

    fn solve_adjoint_in_place(&self, x: &mut [T], hermitian: bool) {
        assert_eq!(
            x.len(),
            self.factors.n,
            "HILU block adjoint solve buffer length mismatch"
        );
        if hermitian {
            self.factors.solve_hermitian_in_place(x);
        } else {
            self.factors.solve_transpose_in_place(x);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Schur-complement coupling (ILUM-style)
// ---------------------------------------------------------------------------

/// Separator coupling data for the Schur-complement (ILUM-style) mode.
///
/// Unknowns are implicitly reordered as `[leaves …, separators …]`:
/// - `e`: separator×leaf coupling block (`E`), rows are separator-local.
/// - `f`: leaf×separator coupling block (`F`), rows are concatenated leaf DOFs.
/// - `sep_factor`: ILU(0) factors of the approximate Schur complement
///   `S ≈ C − E·B⁻¹·F`.
struct SchurCoupling<T: ComplexField> {
    sep_factor: BlockFactors<T>,
    /// Global DOF index per separator-local index.
    sep_indices: Vec<usize>,
    /// Offsets of each leaf block in the concatenated leaf ordering (len = nleaf + 1).
    leaf_base: Vec<usize>,
    /// Global DOF index per concatenated-leaf position.
    leaf_global: Vec<usize>,
    e: CsrMatrix<T>,
    f: CsrMatrix<T>,
}

// ---------------------------------------------------------------------------
// Public: HiluPreconditioner
// ---------------------------------------------------------------------------

/// Hierarchical ILU preconditioner.
///
/// Parallelizes ILU setup by recursively partitioning the matrix via nested
/// dissection. Blocks at each tree level are independent and factorized in
/// parallel. The apply phase is likewise parallel within each level.
pub struct HiluPreconditioner<T: ComplexField> {
    /// Per-block ILU factors. Uncoupled mode: one per dissection node.
    /// Coupled mode: one per leaf; the separator block lives in `coupling`.
    factors: Vec<BlockFactors<T>>,
    /// Global DOF indices per block (mirrors DissectionNode::global_indices).
    block_indices: Vec<Vec<usize>>,
    /// Blocks grouped by level (level 0 = leaves).
    levels: Vec<Vec<usize>>,
    /// Reverse map: global DOF → (block_idx, local_offset).
    /// Retained for future Schur complement extension.
    #[allow(dead_code)]
    global_to_block: Vec<(usize, usize)>,
    /// Schur-complement coupling between leaves and separators (ILUM mode).
    coupling: Option<SchurCoupling<T>>,
    /// Total DOFs.
    n: usize,
}

fn hilu_read_usize(reader: &mut IluCheckpointReader<'_>) -> Result<usize, SolverError> {
    usize::try_from(reader.read_u64()?)
        .map_err(|_| ilu_checkpoint_error("HILU checkpoint integer does not fit usize"))
}

fn hilu_push_pair_vec(output: &mut Vec<u8>, values: &[(usize, usize)]) -> Result<(), SolverError> {
    checkpoint_push_u64(output, values.len())?;
    for &(first, second) in values {
        checkpoint_push_u64(output, first)?;
        checkpoint_push_u64(output, second)?;
    }
    Ok(())
}

fn hilu_read_pair_vec(
    reader: &mut IluCheckpointReader<'_>,
) -> Result<Vec<(usize, usize)>, SolverError> {
    let len = reader.read_len(16)?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push((hilu_read_usize(reader)?, hilu_read_usize(reader)?));
    }
    Ok(values)
}

fn hilu_push_factor<T: ComplexField>(
    output: &mut Vec<u8>,
    factors: &Ilu0Factors<T>,
) -> Result<(), SolverError> {
    validate_checkpoint_factors(factors, None)?;
    checkpoint_push_u64(output, factors.n)?;
    checkpoint_push_usize_vec(output, &factors.l_row_ptrs)?;
    checkpoint_push_u32_vec(output, &factors.l_col_indices)?;
    checkpoint_push_values(output, &factors.l_values)?;
    checkpoint_push_usize_vec(output, &factors.u_row_ptrs)?;
    checkpoint_push_u32_vec(output, &factors.u_col_indices)?;
    checkpoint_push_values(output, &factors.u_values)?;
    checkpoint_push_values(output, &factors.u_diag)?;
    checkpoint_push_values(output, &factors.u_diag_inv)?;
    checkpoint_push_u64(output, factors.diagnostics.skipped_pivots)?;
    checkpoint_push_u64(output, factors.diagnostics.non_finite_entries)?;
    match factors.diagnostics.min_abs_pivot {
        Some(value) => {
            output.push(1);
            checkpoint_push_real(output, value)?;
        }
        None => output.push(0),
    }
    Ok(())
}

fn hilu_read_factor<T: ComplexField>(
    reader: &mut IluCheckpointReader<'_>,
) -> Result<BlockFactors<T>, SolverError> {
    let n = hilu_read_usize(reader)?;
    let factors = Ilu0Factors {
        l_row_ptrs: reader.read_usize_vec()?,
        l_col_indices: reader.read_u32_vec()?,
        l_values: reader.read_values()?,
        u_row_ptrs: reader.read_usize_vec()?,
        u_col_indices: reader.read_u32_vec()?,
        u_values: reader.read_values()?,
        u_diag: reader.read_values()?,
        u_diag_inv: reader.read_values()?,
        n,
        diagnostics: IluDiagnostics {
            skipped_pivots: hilu_read_usize(reader)?,
            non_finite_entries: hilu_read_usize(reader)?,
            min_abs_pivot: match reader.read_u8()? {
                0 => None,
                1 => Some(reader.read_real()?),
                _ => return Err(ilu_checkpoint_error("invalid HILU pivot diagnostic flag")),
            },
        },
    };
    validate_checkpoint_factors(&factors, None)?;
    Ok(BlockFactors { factors })
}

fn hilu_push_csr<T: ComplexField>(
    output: &mut Vec<u8>,
    matrix: &CsrMatrix<T>,
) -> Result<(), SolverError> {
    matrix.validate_canonical_structure()?;
    checkpoint_push_u64(output, matrix.num_rows)?;
    checkpoint_push_u64(output, matrix.num_cols)?;
    checkpoint_push_usize_vec(output, &matrix.row_ptrs)?;
    checkpoint_push_usize_vec(output, &matrix.col_indices)?;
    checkpoint_push_values(output, &matrix.values)?;
    Ok(())
}

fn hilu_read_csr<T: ComplexField>(
    reader: &mut IluCheckpointReader<'_>,
) -> Result<CsrMatrix<T>, SolverError> {
    let rows = hilu_read_usize(reader)?;
    let cols = hilu_read_usize(reader)?;
    let row_ptrs = reader.read_usize_vec()?;
    let col_indices = reader.read_usize_vec()?;
    let values = reader.read_values()?;
    CsrMatrix::try_from_raw_parts(rows, cols, row_ptrs, col_indices, values)
        .map_err(|_| ilu_checkpoint_error("invalid HILU coupling CSR payload"))
}

fn validate_hilu_topology<T: ComplexField>(
    preconditioner: &HiluPreconditioner<T>,
) -> Result<(), SolverError> {
    if preconditioner.n == 0 {
        return Err(ilu_checkpoint_error(
            "HILU checkpoint cannot contain zero-dimensional preconditioner",
        ));
    }
    if preconditioner.factors.is_empty()
        || preconditioner.factors.len() != preconditioner.block_indices.len()
        || preconditioner.global_to_block.len() != preconditioner.n
    {
        return Err(ilu_checkpoint_error(
            "HILU block topology lengths are inconsistent",
        ));
    }

    let mut covered = vec![false; preconditioner.n];
    for (factor, block) in preconditioner
        .factors
        .iter()
        .zip(&preconditioner.block_indices)
    {
        validate_checkpoint_factors(&factor.factors, None)?;
        if factor.factors.n != block.len() || block.is_empty() {
            return Err(ilu_checkpoint_error("HILU block dimension mismatch"));
        }
        for &global in block {
            if global >= preconditioner.n || covered[global] {
                return Err(ilu_checkpoint_error(
                    "HILU block indices are not a partition",
                ));
            }
            covered[global] = true;
        }
    }

    let node_count = preconditioner
        .levels
        .iter()
        .try_fold(0usize, |count, level| count.checked_add(level.len()))
        .ok_or_else(|| ilu_checkpoint_error("HILU level count overflow"))?;
    if preconditioner.levels.is_empty() || node_count == 0 {
        return Err(ilu_checkpoint_error("HILU level topology is empty"));
    }
    // Generated dissection nodes are numbered densely and occur in exactly
    // one level. Bound allocation by the number of stored entries, never by
    // an untrusted maximum ID (which can overflow or request huge storage).
    let mut seen_nodes = vec![false; node_count];
    for level in &preconditioner.levels {
        for &block in level {
            if block >= node_count {
                return Err(ilu_checkpoint_error("HILU level references unknown block"));
            }
            if seen_nodes[block] {
                return Err(ilu_checkpoint_error("HILU level contains duplicate block"));
            }
            seen_nodes[block] = true;
        }
    }

    if let Some(coupling) = &preconditioner.coupling {
        if coupling.sep_indices.is_empty()
            || coupling.leaf_base.len() != preconditioner.factors.len() + 1
            || coupling.leaf_base.first().copied() != Some(0)
            || coupling.leaf_base.windows(2).any(|w| w[0] >= w[1])
            || coupling.leaf_base.last().copied() != Some(coupling.leaf_global.len())
            || coupling.sep_factor.factors.n != coupling.sep_indices.len()
            || coupling.leaf_global.len() != preconditioner.block_indices.iter().map(Vec::len).sum()
        {
            return Err(ilu_checkpoint_error(
                "HILU Schur topology lengths are inconsistent",
            ));
        }
        for (block, window) in preconditioner
            .block_indices
            .iter()
            .zip(coupling.leaf_base.windows(2))
        {
            if coupling.leaf_global[window[0]..window[1]] != block[..] {
                return Err(ilu_checkpoint_error(
                    "HILU leaf ordering does not match block topology",
                ));
            }
        }
        let mut separator_seen = vec![false; preconditioner.n];
        for &global in &coupling.sep_indices {
            if global >= preconditioner.n || covered[global] || separator_seen[global] {
                return Err(ilu_checkpoint_error("HILU separator indices are invalid"));
            }
            separator_seen[global] = true;
            covered[global] = true;
        }
        if covered.iter().any(|&value| !value)
            || coupling.e.num_rows != coupling.sep_indices.len()
            || coupling.e.num_cols != coupling.leaf_global.len()
            || coupling.f.num_rows != coupling.leaf_global.len()
            || coupling.f.num_cols != coupling.sep_indices.len()
        {
            return Err(ilu_checkpoint_error(
                "HILU Schur coupling dimensions are invalid",
            ));
        }
        validate_checkpoint_factors(&coupling.sep_factor.factors, None)?;
        coupling.e.validate_canonical_structure()?;
        coupling.f.validate_canonical_structure()?;
    } else if covered.iter().any(|&value| !value) {
        return Err(ilu_checkpoint_error(
            "HILU blocks do not cover all matrix DOFs",
        ));
    }

    if preconditioner.coupling.is_none() {
        // In uncoupled mode every tree node is represented by one block
        // factor, so the reverse map can and should be checked exactly.
        // Merely checking that IDs are below `n` allowed a corrupted but
        // otherwise valid checkpoint to restore a map pointing at the wrong
        // factor or local row.
        let mut expected = vec![None; preconditioner.n];
        for (block, indices) in preconditioner.block_indices.iter().enumerate() {
            for (local, &global) in indices.iter().enumerate() {
                expected[global] = Some((block, local));
            }
        }
        if preconditioner.global_to_block.len() != expected.len()
            || preconditioner
                .global_to_block
                .iter()
                .zip(expected)
                .any(|(&actual, expected)| Some(actual) != expected)
        {
            return Err(ilu_checkpoint_error("HILU reverse block map is invalid"));
        }
    } else if preconditioner
        .global_to_block
        .iter()
        .any(|&(block, local)| block >= node_count || local >= preconditioner.n)
    {
        // Coupled checkpoints retain separator nodes in this map, while
        // only leaf blocks are stored in `block_indices`.
        return Err(ilu_checkpoint_error("HILU reverse block map is invalid"));
    }
    Ok(())
}

/// Build the nested-dissection tree for `matrix` and group nodes by level.
fn build_dissection<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: &HiluConfig,
) -> (Vec<DissectionNode>, Vec<Vec<usize>>) {
    let n = matrix.num_rows;

    // Auto-compute max levels
    let max_levels = config.max_levels.unwrap_or_else(|| {
        if n <= config.min_block_size {
            0
        } else {
            ((n as f64) / (config.min_block_size as f64)).log2().ceil() as usize
        }
    });

    // Build adjacency graph (symmetric structure)
    let adjacency = build_adjacency(matrix);

    // Nested dissection
    let all_indices: Vec<usize> = (0..n).collect();
    let mut nodes: Vec<DissectionNode> = Vec::new();
    nested_dissection(
        &adjacency,
        &all_indices,
        config.min_block_size,
        max_levels,
        0,
        &mut nodes,
    );

    // Group nodes by level
    let max_level = nodes.iter().map(|nd| nd.level).max().unwrap_or(0);
    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (idx, nd) in nodes.iter().enumerate() {
        levels[nd.level].push(idx);
    }

    (nodes, levels)
}

/// Factorize a set of blocks in parallel (one ILU(0) per block).
fn factor_blocks<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    block_indices: &[Vec<usize>],
) -> Vec<BlockFactors<T>> {
    use rayon::prelude::*;
    let num_blocks = block_indices.len();
    let mut factors: Vec<Option<BlockFactors<T>>> = (0..num_blocks).map(|_| None).collect();
    if num_blocks > 1 {
        let factor_results: Vec<(usize, BlockFactors<T>)> = (0..num_blocks)
            .into_par_iter()
            .map(|blk| {
                let sub = extract_submatrix(matrix, &block_indices[blk]);
                (blk, BlockFactors::from_local_csr(&sub))
            })
            .collect();
        for (blk, f) in factor_results {
            factors[blk] = Some(f);
        }
    } else {
        for (blk, indices) in block_indices.iter().enumerate() {
            let sub = extract_submatrix(matrix, indices);
            factors[blk] = Some(BlockFactors::from_local_csr(&sub));
        }
    }
    factors.into_iter().map(|f| f.unwrap()).collect()
}

impl<T: ComplexField> HiluPreconditioner<T> {
    /// Serialize the complete H-ILU factorization and block topology.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate_hilu_topology(self)?;
        let mut output = Vec::new();
        output.extend_from_slice(HILU_CHECKPOINT_MAGIC);
        output.extend_from_slice(&HILU_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(u8::from(
            std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>(),
        ));
        output.push(0);

        checkpoint_push_u64(&mut output, self.n)?;
        checkpoint_push_u64(&mut output, self.factors.len())?;
        for factor in &self.factors {
            hilu_push_factor(&mut output, &factor.factors)?;
        }
        checkpoint_push_u64(&mut output, self.block_indices.len())?;
        for block in &self.block_indices {
            checkpoint_push_usize_vec(&mut output, block)?;
        }
        checkpoint_push_u64(&mut output, self.levels.len())?;
        for level in &self.levels {
            checkpoint_push_usize_vec(&mut output, level)?;
        }
        hilu_push_pair_vec(&mut output, &self.global_to_block)?;
        match &self.coupling {
            None => output.push(0),
            Some(coupling) => {
                output.push(1);
                hilu_push_factor(&mut output, &coupling.sep_factor.factors)?;
                checkpoint_push_usize_vec(&mut output, &coupling.sep_indices)?;
                checkpoint_push_usize_vec(&mut output, &coupling.leaf_base)?;
                checkpoint_push_usize_vec(&mut output, &coupling.leaf_global)?;
                hilu_push_csr(&mut output, &coupling.e)?;
                hilu_push_csr(&mut output, &coupling.f)?;
            }
        }
        Ok(output)
    }

    /// Restore an H-ILU factorization produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = IluCheckpointReader::new(bytes);
        if reader.take::<8>()? != *HILU_CHECKPOINT_MAGIC {
            return Err(ilu_checkpoint_error("bad HILU checkpoint magic"));
        }
        if reader.read_u16()? != HILU_CHECKPOINT_FORMAT_VERSION {
            return Err(ilu_checkpoint_error("unsupported HILU checkpoint version"));
        }
        if reader.read_u8()? as usize != std::mem::size_of::<T>()
            || reader.read_u8()? as usize != std::mem::size_of::<T::Real>()
            || reader.read_u8()?
                != u8::from(std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>())
        {
            return Err(ilu_checkpoint_error(
                "HILU checkpoint scalar metadata does not match requested type",
            ));
        }
        if reader.read_u8()? != 0 {
            return Err(ilu_checkpoint_error("unsupported HILU checkpoint flags"));
        }

        let n = hilu_read_usize(&mut reader)?;
        if n == 0 || n > bytes.len() {
            return Err(ilu_checkpoint_error(
                "HILU checkpoint dimension is outside payload bounds",
            ));
        }
        let factor_count = reader.read_len(1)?;
        if factor_count == 0 {
            return Err(ilu_checkpoint_error("HILU checkpoint has no leaf factors"));
        }
        let mut factors = Vec::with_capacity(factor_count);
        for _ in 0..factor_count {
            factors.push(hilu_read_factor(&mut reader)?);
        }
        let block_count = reader.read_len(1)?;
        if block_count != factor_count {
            return Err(ilu_checkpoint_error(
                "HILU block and factor counts do not match",
            ));
        }
        let mut block_indices = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            block_indices.push(reader.read_usize_vec()?);
        }
        let level_count = reader.read_len(1)?;
        let mut levels = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            levels.push(reader.read_usize_vec()?);
        }
        let global_to_block = hilu_read_pair_vec(&mut reader)?;
        let coupling = match reader.read_u8()? {
            0 => None,
            1 => Some(SchurCoupling {
                sep_factor: hilu_read_factor(&mut reader)?,
                sep_indices: reader.read_usize_vec()?,
                leaf_base: reader.read_usize_vec()?,
                leaf_global: reader.read_usize_vec()?,
                e: hilu_read_csr(&mut reader)?,
                f: hilu_read_csr(&mut reader)?,
            }),
            _ => return Err(ilu_checkpoint_error("invalid HILU coupling marker")),
        };
        if !reader.is_at_end() {
            return Err(ilu_checkpoint_error("trailing HILU checkpoint bytes"));
        }
        let preconditioner = Self {
            factors,
            block_indices,
            levels,
            global_to_block,
            coupling,
            n,
        };
        validate_hilu_topology(&preconditioner)?;
        Ok(preconditioner)
    }

    /// Build the H-ILU preconditioner from a CSR matrix with default config.
    ///
    /// This is the coupled (ILUM-style) mode with Schur-complement coupling;
    /// see [`HiluPreconditioner::from_csr_coupled`].
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr_coupled(matrix)
    }

    /// Fallible H-ILU construction with the default coupled configuration.
    ///
    /// This rejects malformed/non-finite input and any block whose canonical
    /// ILU(0) setup skipped or regularized a pivot. The historical
    /// [`Self::from_csr`] constructor retains its compatibility behavior.
    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        Self::try_from_csr_coupled_with_config(matrix, &HiluConfig::default(), 0.0)
    }

    /// Build the H-ILU preconditioner from a CSR matrix with custom config.
    ///
    /// This is the coupled (ILUM-style) mode with Schur-complement coupling
    /// and no Schur dropping; see
    /// [`HiluPreconditioner::from_csr_coupled_with_config`]. For the
    /// historical uncoupled (block-Jacobi-like) mode, use
    /// [`HiluPreconditioner::from_csr_uncoupled_with_config`].
    pub fn from_csr_with_config(matrix: &CsrMatrix<T>, config: &HiluConfig) -> Self {
        Self::from_csr_coupled_with_config(matrix, config, 0.0)
    }

    /// Fallible H-ILU construction with custom coupled configuration.
    pub fn try_from_csr_with_config(
        matrix: &CsrMatrix<T>,
        config: &HiluConfig,
    ) -> Result<Self, SolverError> {
        Self::try_from_csr_coupled_with_config(matrix, config, 0.0)
    }

    /// Build the historical *uncoupled* H-ILU with default config.
    ///
    /// Every tree node (leaves and separators) is factored independently and
    /// the coupling blocks are discarded, which behaves like block Jacobi.
    /// Prefer the default [`HiluPreconditioner::from_csr`] (coupled mode)
    /// for much better convergence.
    pub fn from_csr_uncoupled(matrix: &CsrMatrix<T>) -> Self {
        Self::build_uncoupled(matrix, &HiluConfig::default())
    }

    /// Fallible historical uncoupled H-ILU construction.
    pub fn try_from_csr_uncoupled(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        Self::try_from_csr_uncoupled_with_config(matrix, &HiluConfig::default())
    }

    /// Build the historical *uncoupled* H-ILU with custom config.
    ///
    /// See [`HiluPreconditioner::from_csr_uncoupled`].
    pub fn from_csr_uncoupled_with_config(matrix: &CsrMatrix<T>, config: &HiluConfig) -> Self {
        Self::build_uncoupled(matrix, config)
    }

    /// Fallible historical uncoupled H-ILU construction with custom config.
    pub fn try_from_csr_uncoupled_with_config(
        matrix: &CsrMatrix<T>,
        config: &HiluConfig,
    ) -> Result<Self, SolverError> {
        Self::validate_setup_input(matrix, config)?;
        Self::check_diagnostics(Self::build_uncoupled(matrix, config))
    }

    /// Uncoupled construction: one independent ILU(0) per dissection node,
    /// coupling blocks discarded.
    fn build_uncoupled(matrix: &CsrMatrix<T>, config: &HiluConfig) -> Self {
        let n = matrix.num_rows;
        let (nodes, levels) = build_dissection(matrix, config);

        // Build global-to-block reverse map
        let mut global_to_block = vec![(0usize, 0usize); n];
        for (blk, nd) in nodes.iter().enumerate() {
            for (local, &global) in nd.global_indices.iter().enumerate() {
                global_to_block[global] = (blk, local);
            }
        }

        // Extract submatrices and factorize each block.
        let block_indices: Vec<Vec<usize>> =
            nodes.iter().map(|nd| nd.global_indices.clone()).collect();
        debug_assert_block_partition(n, &block_indices);
        let factors = factor_blocks(matrix, &block_indices);

        Self {
            factors,
            block_indices,
            levels,
            global_to_block,
            coupling: None,
            n,
        }
    }

    /// Build the ILUM-style H-ILU preconditioner with Schur-complement
    /// coupling, using default config and no Schur dropping.
    ///
    /// This is the same construction as the default
    /// [`HiluPreconditioner::from_csr`]: the separator/coupling blocks
    /// are kept: leaf blocks are factored with ILU(0), the approximate Schur
    /// complement `S ≈ C − E·B⁻¹·F` on the separators is formed and factored,
    /// and `apply` performs a 2×2 block forward/backward substitution. This
    /// converges in markedly fewer Krylov iterations and degrades much more
    /// slowly as the block count grows.
    pub fn from_csr_coupled(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr_coupled_with_config(matrix, &HiluConfig::default(), 0.0)
    }

    /// Fallible coupled H-ILU construction with default configuration.
    pub fn try_from_csr_coupled(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        Self::try_from_csr_coupled_with_config(matrix, &HiluConfig::default(), 0.0)
    }

    /// Build the ILUM-style H-ILU preconditioner with custom config.
    ///
    /// `schur_drop_tol` is a relative dropping tolerance applied to *new*
    /// fill in the approximate Schur complement: entries not present in the
    /// original separator block `C` are dropped when
    /// `|s_ij| < schur_drop_tol * ‖C(i,:)‖₂`. Entries of the original
    /// separator pattern and the diagonal are always kept (the same
    /// pattern-preserving discipline as the leaf ILU(0)). `0.0` keeps all
    /// computed fill and is the most accurate option.
    pub fn from_csr_coupled_with_config(
        matrix: &CsrMatrix<T>,
        config: &HiluConfig,
        schur_drop_tol: f64,
    ) -> Self {
        let n = matrix.num_rows;
        let (nodes, levels) = build_dissection(matrix, config);

        // Leaves (interior blocks) vs. internal nodes (separators).
        let leaf_nodes: Vec<usize> = (0..nodes.len())
            .filter(|&i| nodes[i].left.is_none())
            .collect();
        let sep_nodes: Vec<usize> = (0..nodes.len())
            .filter(|&i| nodes[i].left.is_some())
            .collect();

        // Degenerate case: no separators → identical to the uncoupled build.
        if sep_nodes.is_empty() {
            return Self::build_uncoupled(matrix, config);
        }

        // Build global-to-block reverse map
        let mut global_to_block = vec![(0usize, 0usize); n];
        for (blk, nd) in nodes.iter().enumerate() {
            for (local, &global) in nd.global_indices.iter().enumerate() {
                global_to_block[global] = (blk, local);
            }
        }

        // Separator-local numbering.
        let mut sep_of_global = vec![usize::MAX; n];
        let mut sep_indices: Vec<usize> = Vec::new();
        for &sn in &sep_nodes {
            for &g in &nodes[sn].global_indices {
                sep_of_global[g] = sep_indices.len();
                sep_indices.push(g);
            }
        }
        let nsep = sep_indices.len();

        // Concatenated leaf ordering.
        let block_indices: Vec<Vec<usize>> = leaf_nodes
            .iter()
            .map(|&l| nodes[l].global_indices.clone())
            .collect();
        let nleaf = block_indices.len();
        let mut leaf_of_global = vec![usize::MAX; n];
        let mut local_of_global = vec![0usize; n];
        let mut leaf_base = vec![0usize];
        let mut leaf_global: Vec<usize> = Vec::new();
        for (l, indices) in block_indices.iter().enumerate() {
            for &g in indices {
                leaf_of_global[g] = l;
                local_of_global[g] = leaf_global.len() - leaf_base[l];
                leaf_global.push(g);
            }
            leaf_base.push(leaf_global.len());
        }
        let nlt = leaf_global.len();

        // ILU(0) on each leaf block (parallel).
        let factors = factor_blocks(matrix, &block_indices);

        // F: rows = concatenated leaf DOFs, cols = separator-local.
        let mut f_values = Vec::new();
        let mut f_cols = Vec::new();
        let mut f_ptrs = vec![0usize];
        for &g in &leaf_global {
            for idx in matrix.row_ptrs[g]..matrix.row_ptrs[g + 1] {
                let s = sep_of_global[matrix.col_indices[idx]];
                if s != usize::MAX {
                    f_values.push(matrix.values[idx]);
                    f_cols.push(s);
                }
            }
            f_ptrs.push(f_values.len());
        }

        // E: rows = separator-local, cols = concatenated leaf position.
        // Entries are additionally bucketed per leaf (grouped by separator row)
        // for the Schur update below.
        let mut e_values = Vec::new();
        let mut e_cols = Vec::new();
        let mut e_ptrs = vec![0usize];
        let mut e_by_leaf: Vec<Vec<(usize, usize, T)>> = (0..nleaf).map(|_| Vec::new()).collect();
        for (s, &g) in sep_indices.iter().enumerate() {
            for idx in matrix.row_ptrs[g]..matrix.row_ptrs[g + 1] {
                let c = matrix.col_indices[idx];
                let l = leaf_of_global[c];
                if l != usize::MAX {
                    let v = matrix.values[idx];
                    e_values.push(v);
                    e_cols.push(leaf_base[l] + local_of_global[c]);
                    e_by_leaf[l].push((s, local_of_global[c], v));
                }
            }
            e_ptrs.push(e_values.len());
        }
        let e = CsrMatrix::from_raw_parts(nsep, nlt, e_ptrs, e_cols, e_values);
        let f = CsrMatrix::from_raw_parts(nlt, nsep, f_ptrs, f_cols, f_values);

        // Initialize the Schur complement rows with C = A(sep, sep), keeping
        // track of the original pattern and row norms for dropping.
        let mut s_rows: Vec<BTreeMap<usize, T>> = (0..nsep).map(|_| BTreeMap::new()).collect();
        let mut c_pattern: Vec<BTreeSet<usize>> = (0..nsep).map(|_| BTreeSet::new()).collect();
        let mut c_norms = vec![T::Real::zero(); nsep];
        for (s, &g) in sep_indices.iter().enumerate() {
            let mut scale = T::Real::zero();
            let mut sum_squares = T::Real::one();
            for idx in matrix.row_ptrs[g]..matrix.row_ptrs[g + 1] {
                let t = sep_of_global[matrix.col_indices[idx]];
                if t != usize::MAX {
                    let v = matrix.values[idx];
                    update_scaled_norm(&mut scale, &mut sum_squares, v.norm());
                    s_rows[s].insert(t, v);
                    c_pattern[s].insert(t);
                }
            }
            c_norms[s] = finish_scaled_norm(scale, sum_squares);
        }

        // Schur update: S -= E_ℓ · B_ℓ⁻¹ · F_ℓ for each leaf ℓ.
        // For every separator column t coupled to leaf ℓ, solve
        // B_ℓ w = F_ℓ(:, t) and accumulate S(:, t) -= E_ℓ w.
        for (l, factor) in factors.iter().enumerate() {
            let lo = leaf_base[l];
            let hi = leaf_base[l + 1];
            let nl = hi - lo;

            // Columns of F restricted to this leaf, transposed: sep t → [(k, v)].
            let mut f_cols_t: BTreeMap<usize, Vec<(usize, T)>> = BTreeMap::new();
            for row in lo..hi {
                for idx in f.row_ptrs[row]..f.row_ptrs[row + 1] {
                    f_cols_t
                        .entry(f.col_indices[idx])
                        .or_default()
                        .push((row - lo, f.values[idx]));
                }
            }
            if f_cols_t.is_empty() {
                continue;
            }

            let e_l = &e_by_leaf[l]; // grouped by separator row, ascending
            for (t, col_entries) in &f_cols_t {
                let mut w = vec![T::zero(); nl];
                for &(k, v) in col_entries {
                    w[k] = v;
                }
                factor.solve_in_place(&mut w);

                let mut i = 0;
                while i < e_l.len() {
                    let s = e_l[i].0;
                    let mut acc = T::zero();
                    while i < e_l.len() && e_l[i].0 == s {
                        acc += e_l[i].2 * w[e_l[i].1];
                        i += 1;
                    }
                    if !acc.norm().is_zero() {
                        let entry = s_rows[s].entry(*t).or_insert_with(T::zero);
                        *entry -= acc;
                    }
                }
            }
        }

        // Convert S to CSR with dropping: always keep the diagonal and the
        // original C pattern; keep new fill above the relative threshold.
        let drop_tol = T::Real::from_f64(schur_drop_tol).unwrap();
        let mut s_values = Vec::new();
        let mut s_cols = Vec::new();
        let mut s_ptrs = vec![0usize];
        for (s, row) in s_rows.iter().enumerate() {
            let norm = c_norms[s].max(T::Real::from_f64(1e-30).unwrap());
            for (&t, &v) in row {
                if v.norm().is_zero() {
                    continue;
                }
                let keep = t == s || c_pattern[s].contains(&t) || v.norm() >= drop_tol * norm;
                if keep {
                    s_values.push(v);
                    s_cols.push(t);
                }
            }
            s_ptrs.push(s_values.len());
        }
        let schur = CsrMatrix::from_raw_parts(nsep, nsep, s_ptrs, s_cols, s_values);
        let sep_factor = BlockFactors::from_local_csr(&schur);

        Self {
            factors,
            block_indices,
            levels,
            global_to_block,
            coupling: Some(SchurCoupling {
                sep_factor,
                sep_indices,
                leaf_base,
                leaf_global,
                e,
                f,
            }),
            n,
        }
    }

    /// Fallible coupled H-ILU construction with an explicit Schur drop
    /// tolerance.
    pub fn try_from_csr_coupled_with_config(
        matrix: &CsrMatrix<T>,
        config: &HiluConfig,
        schur_drop_tol: f64,
    ) -> Result<Self, SolverError> {
        Self::validate_setup_input(matrix, config)?;
        if !schur_drop_tol.is_finite() || schur_drop_tol < 0.0 {
            return Err(SolverError::OperationFailed {
                operation: "HILU setup",
                reason: "schur_drop_tol must be finite and non-negative".to_string(),
            });
        }
        Self::check_diagnostics(Self::from_csr_coupled_with_config(
            matrix,
            config,
            schur_drop_tol,
        ))
    }

    fn validate_setup_input(matrix: &CsrMatrix<T>, config: &HiluConfig) -> Result<(), SolverError> {
        matrix.validate_canonical_structure()?;
        if matrix.num_rows != matrix.num_cols {
            return Err(SolverError::OperationFailed {
                operation: "HILU setup",
                reason: format!(
                    "matrix must be square, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        if config.min_block_size == 0 {
            return Err(SolverError::OperationFailed {
                operation: "HILU setup",
                reason: "min_block_size must be greater than zero".to_string(),
            });
        }
        if matrix.values.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "HILU setup",
                reason: "matrix contains NaN or infinity".to_string(),
            });
        }
        CsrMatrix::try_from_raw_parts(
            matrix.num_rows,
            matrix.num_cols,
            matrix.row_ptrs.clone(),
            matrix.col_indices.clone(),
            matrix.values.clone(),
        )
        .map(|_| ())
    }

    fn check_diagnostics(preconditioner: Self) -> Result<Self, SolverError> {
        let mut skipped_pivots = 0usize;
        let mut non_finite_entries = 0usize;
        for block in &preconditioner.factors {
            skipped_pivots += block.factors.diagnostics.skipped_pivots;
            non_finite_entries += block.factors.diagnostics.non_finite_entries;
        }
        if let Some(coupling) = &preconditioner.coupling {
            skipped_pivots += coupling.sep_factor.factors.diagnostics.skipped_pivots;
            non_finite_entries += coupling.sep_factor.factors.diagnostics.non_finite_entries;
        }
        if non_finite_entries != 0 {
            return Err(SolverError::OperationFailed {
                operation: "HILU setup",
                reason: format!("{non_finite_entries} non-finite matrix entries"),
            });
        }
        if skipped_pivots != 0 {
            return Err(SolverError::OperationFailed {
                operation: "HILU setup",
                reason: format!(
                    "{skipped_pivots} block pivots were below the configured relative threshold"
                ),
            });
        }
        Ok(preconditioner)
    }

    /// Number of blocks in the dissection tree (leaf blocks in coupled mode).
    pub fn num_blocks(&self) -> usize {
        self.factors.len()
    }

    /// Number of levels in the dissection tree.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Whether this preconditioner couples leaves and separators through the
    /// approximate Schur complement (ILUM-style), as opposed to the
    /// uncoupled block-Jacobi-like mode.
    pub fn is_coupled(&self) -> bool {
        self.coupling.is_some()
    }

    /// Aggregate setup diagnostics from every block and, when present, the
    /// Schur-complement factor. Checked constructors reject non-zero failure
    /// counters; compatibility callers can inspect the retained policy.
    pub fn diagnostics(&self) -> IluDiagnostics<T::Real> {
        let mut aggregate: IluDiagnostics<T::Real> = IluDiagnostics {
            skipped_pivots: 0,
            non_finite_entries: 0,
            min_abs_pivot: None,
        };
        let mut include = |diagnostics: &IluDiagnostics<T::Real>| {
            aggregate.skipped_pivots += diagnostics.skipped_pivots;
            aggregate.non_finite_entries += diagnostics.non_finite_entries;
            if let Some(pivot) = diagnostics.min_abs_pivot {
                aggregate.min_abs_pivot = Some(match aggregate.min_abs_pivot {
                    Some(current) => current.min(pivot),
                    None => pivot,
                });
            }
        };
        for block in &self.factors {
            include(&block.factors.diagnostics);
        }
        if let Some(coupling) = &self.coupling {
            include(&coupling.sep_factor.factors.diagnostics);
        }
        aggregate
    }

    /// Solve `B_ℓ x_ℓ = rhs_ℓ` for every leaf block, in the concatenated
    /// leaf ordering. Parallel over leaves for large problems.
    fn leaf_solve(&self, rhs: &[T], out: &mut [T]) {
        let coupling = self
            .coupling
            .as_ref()
            .expect("leaf_solve requires coupling");
        let leaf_base = &coupling.leaf_base;
        let nleaf = self.factors.len();
        if nleaf > 1 && self.n >= 1000 {
            use rayon::prelude::*;
            #[cfg(debug_assertions)]
            debug_assert_partition(leaf_base, out.len());
            let out_ptr = DisjointScatterPtr(out.as_mut_ptr());
            (0..nleaf)
                .into_par_iter()
                .for_each_init(Vec::<T>::new, |buf, l| {
                    let lo = leaf_base[l];
                    let hi = leaf_base[l + 1];
                    buf.clear();
                    buf.extend_from_slice(&rhs[lo..hi]);
                    self.factors[l].solve_in_place(buf);
                    for (i, &v) in buf.iter().enumerate() {
                        unsafe {
                            out_ptr.write(lo + i, v);
                        }
                    }
                });
        } else {
            for l in 0..nleaf {
                let lo = leaf_base[l];
                let hi = leaf_base[l + 1];
                out[lo..hi].copy_from_slice(&rhs[lo..hi]);
                self.factors[l].solve_in_place(&mut out[lo..hi]);
            }
        }
    }

    /// Solve the leaf block diagonal with the requested transpose action.
    fn leaf_adjoint_solve(&self, rhs: &[T], out: &mut [T], hermitian: bool) {
        let coupling = self
            .coupling
            .as_ref()
            .expect("leaf_adjoint_solve requires coupling");
        let leaf_base = &coupling.leaf_base;
        let nleaf = self.factors.len();
        if nleaf > 1 && self.n >= 1000 {
            use rayon::prelude::*;
            #[cfg(debug_assertions)]
            debug_assert_partition(leaf_base, out.len());
            let out_ptr = DisjointScatterPtr(out.as_mut_ptr());
            (0..nleaf)
                .into_par_iter()
                .for_each_init(Vec::<T>::new, |buf, l| {
                    let lo = leaf_base[l];
                    let hi = leaf_base[l + 1];
                    buf.clear();
                    buf.extend_from_slice(&rhs[lo..hi]);
                    self.factors[l].solve_adjoint_in_place(buf, hermitian);
                    for (i, &value) in buf.iter().enumerate() {
                        unsafe {
                            out_ptr.write(lo + i, value);
                        }
                    }
                });
        } else {
            for l in 0..nleaf {
                let lo = leaf_base[l];
                let hi = leaf_base[l + 1];
                out[lo..hi].copy_from_slice(&rhs[lo..hi]);
                self.factors[l].solve_adjoint_in_place(&mut out[lo..hi], hermitian);
            }
        }
    }

    /// Coupled apply: 2×2 block forward/backward substitution on
    /// `[B F; E C]` using the factored leaves and the factored approximate
    /// Schur complement.
    fn apply_coupled(&self, r: &[T], result: &mut [T], coupling: &SchurCoupling<T>) {
        let nlt = coupling.leaf_global.len();

        // Forward: y1 = B⁻¹ r1 (per leaf).
        let mut rl = vec![T::zero(); nlt];
        for (i, &g) in coupling.leaf_global.iter().enumerate() {
            rl[i] = r[g];
        }
        let mut yl = vec![T::zero(); nlt];
        self.leaf_solve(&rl, &mut yl);

        // Separator RHS update: r2' = r2 − E·y1.
        let mut rs: Vec<T> = coupling.sep_indices.iter().map(|&g| r[g]).collect();
        for (s, slot) in rs.iter_mut().enumerate() {
            let mut acc = T::zero();
            for idx in coupling.e.row_ptrs[s]..coupling.e.row_ptrs[s + 1] {
                acc += coupling.e.values[idx] * yl[coupling.e.col_indices[idx]];
            }
            *slot -= acc;
        }

        // Separator solve: z2 = S⁻¹ r2'.
        coupling.sep_factor.solve_in_place(&mut rs); // rs now holds z2

        // Back-substitution: y1 ← y1 − B⁻¹·F·z2.
        let mut t = vec![T::zero(); nlt];
        for (row, slot) in t.iter_mut().enumerate() {
            let mut acc = T::zero();
            for idx in coupling.f.row_ptrs[row]..coupling.f.row_ptrs[row + 1] {
                acc += coupling.f.values[idx] * rs[coupling.f.col_indices[idx]];
            }
            *slot = acc;
        }
        let mut w = vec![T::zero(); nlt];
        self.leaf_solve(&t, &mut w);
        for i in 0..nlt {
            yl[i] -= w[i];
        }

        // Scatter to the global ordering.
        for (i, &g) in coupling.leaf_global.iter().enumerate() {
            result[g] = yl[i];
        }
        for (s, &g) in coupling.sep_indices.iter().enumerate() {
            result[g] = rs[s];
        }
    }

    /// Apply the exact transpose or Hermitian action of the coupled block
    /// preconditioner. For `M = [B F; E C]` and
    /// `S = C - E B⁻¹ F`, block elimination reduces the adjoint solve to
    /// the stored leaf and Schur factors.
    fn apply_coupled_adjoint(&self, r: &[T], result: &mut [T], hermitian: bool) {
        let coupling = self
            .coupling
            .as_ref()
            .expect("apply_coupled_adjoint requires coupling");
        let nsep = coupling.sep_indices.len();
        let nlt = coupling.leaf_global.len();
        let adjoint = |value: T| if hermitian { value.conj() } else { value };

        let mut leaf_rhs = vec![T::zero(); nlt];
        for (local, &global) in coupling.leaf_global.iter().enumerate() {
            leaf_rhs[local] = r[global];
        }
        let mut sep_rhs = vec![T::zero(); nsep];
        for (local, &global) in coupling.sep_indices.iter().enumerate() {
            sep_rhs[local] = r[global];
        }

        // (Cᴴ - Fᴴ Bᴴ⁻¹ Eᴴ) x_s = r_s - Fᴴ Bᴴ⁻¹ r_l.
        let mut leaf_solution = vec![T::zero(); nlt];
        self.leaf_adjoint_solve(&leaf_rhs, &mut leaf_solution, hermitian);
        for (row, &value) in leaf_solution.iter().enumerate() {
            for idx in coupling.f.row_ptrs[row]..coupling.f.row_ptrs[row + 1] {
                let separator = coupling.f.col_indices[idx];
                sep_rhs[separator] -= adjoint(coupling.f.values[idx]) * value;
            }
        }
        coupling
            .sep_factor
            .solve_adjoint_in_place(&mut sep_rhs, hermitian);

        // Bᴴ x_l = r_l - Eᴴ x_s.
        let mut recovered_leaf_rhs = leaf_rhs;
        for (row, &value) in sep_rhs.iter().enumerate() {
            for idx in coupling.e.row_ptrs[row]..coupling.e.row_ptrs[row + 1] {
                let leaf = coupling.e.col_indices[idx];
                recovered_leaf_rhs[leaf] -= adjoint(coupling.e.values[idx]) * value;
            }
        }
        self.leaf_adjoint_solve(&recovered_leaf_rhs, &mut leaf_solution, hermitian);

        for (local, &global) in coupling.leaf_global.iter().enumerate() {
            result[global] = leaf_solution[local];
        }
        for (local, &global) in coupling.sep_indices.iter().enumerate() {
            result[global] = sep_rhs[local];
        }
    }
}

impl<T: ComplexField> Preconditioner<T> for HiluPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut result = vec![T::zero(); self.n];
        self.apply_into(r, &mut result);
        result
    }

    fn apply_into(&self, r: &[T], result: &mut [T]) {
        assert_eq!(r.len(), self.n, "HILU RHS length mismatch");
        assert_eq!(result.len(), self.n, "HILU output length mismatch");
        result.fill(T::zero());

        // Coupled (ILUM-style) mode: 2×2 block forward/backward substitution.
        if let Some(coupling) = &self.coupling {
            self.apply_coupled(r, result, coupling);
            return;
        }

        // Solve each disjoint block independently and scatter results.
        {
            use rayon::prelude::*;
            if self.factors.len() > 1 && self.n >= 1000 {
                // Debug-only disjointness check: block index sets must be
                // pairwise non-overlapping and in bounds, else the scatter
                // below is a data race.
                debug_assert!({
                    let mut seen = vec![false; result.len()];
                    self.block_indices.iter().all(|indices| {
                        indices.iter().all(|&g| {
                            if g >= seen.len() || seen[g] {
                                return false;
                            }
                            seen[g] = true;
                            true
                        })
                    })
                });
                let result_ptr = DisjointScatterPtr(result.as_mut_ptr());
                (0..self.factors.len()).into_par_iter().for_each_init(
                    Vec::<T>::new,
                    |local_x, blk| {
                        let indices = &self.block_indices[blk];
                        local_x.resize(indices.len(), T::zero());
                        for (local, &global) in indices.iter().enumerate() {
                            local_x[local] = r[global];
                        }
                        self.factors[blk].solve_in_place(local_x);
                        for (local, &global) in indices.iter().enumerate() {
                            unsafe {
                                result_ptr.write(global, local_x[local]);
                            }
                        }
                    },
                );
            } else {
                let max_local_n = self.block_indices.iter().map(Vec::len).max().unwrap_or(0);
                let mut local_x = vec![T::zero(); max_local_n];

                for (blk, indices) in self.block_indices.iter().enumerate() {
                    let local_n = indices.len();
                    for (local, &global) in indices.iter().enumerate() {
                        local_x[local] = r[global];
                    }
                    self.factors[blk].solve_in_place(&mut local_x[..local_n]);
                    for (local, &global) in indices.iter().enumerate() {
                        result[global] = local_x[local];
                    }
                }
            }
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        let mut result = vec![T::zero(); self.n];
        if self.coupling.is_some() {
            self.apply_coupled_adjoint(r, &mut result, false);
        } else {
            self.apply_adjoint_uncoupled(r, &mut result, false);
        }
        result
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], result: &mut [T]) {
        if self.coupling.is_some() {
            self.apply_coupled_adjoint(r, result, false);
        } else {
            self.apply_adjoint_uncoupled(r, result, false);
        }
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut result = vec![T::zero(); self.n];
        if self.coupling.is_some() {
            self.apply_coupled_adjoint(r, &mut result, true);
        } else {
            self.apply_adjoint_uncoupled(r, &mut result, true);
        }
        result
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], result: &mut [T]) {
        if self.coupling.is_some() {
            self.apply_coupled_adjoint(r, result, true);
        } else {
            self.apply_adjoint_uncoupled(r, result, true);
        }
    }
}

impl<T: ComplexField> HiluPreconditioner<T> {
    fn apply_adjoint_uncoupled(&self, r: &[T], result: &mut [T], hermitian: bool) {
        assert_eq!(r.len(), self.n, "HILU adjoint RHS length mismatch");
        assert_eq!(result.len(), self.n, "HILU adjoint output length mismatch");
        assert!(
            self.coupling.is_none(),
            "HILU coupled Schur-complement adjoint action is not implemented"
        );
        result.fill(T::zero());

        use rayon::prelude::*;
        if self.factors.len() > 1 && self.n >= 1000 {
            // Debug-only disjointness check: block index sets must be
            // pairwise non-overlapping and in bounds, else the scatter
            // below is a data race.
            debug_assert!({
                let mut seen = vec![false; result.len()];
                self.block_indices.iter().all(|indices| {
                    indices.iter().all(|&g| {
                        if g >= seen.len() || seen[g] {
                            return false;
                        }
                        seen[g] = true;
                        true
                    })
                })
            });
            let result_ptr = DisjointScatterPtr(result.as_mut_ptr());
            (0..self.factors.len())
                .into_par_iter()
                .for_each_init(Vec::<T>::new, |local_x, blk| {
                    let indices = &self.block_indices[blk];
                    local_x.resize(indices.len(), T::zero());
                    for (local, &global) in indices.iter().enumerate() {
                        local_x[local] = r[global];
                    }
                    self.factors[blk].solve_adjoint_in_place(local_x, hermitian);
                    for (local, &global) in indices.iter().enumerate() {
                        unsafe {
                            result_ptr.write(global, local_x[local]);
                        }
                    }
                });
        } else {
            let max_local_n = self.block_indices.iter().map(Vec::len).max().unwrap_or(0);
            let mut local_x = vec![T::zero(); max_local_n];
            for (blk, indices) in self.block_indices.iter().enumerate() {
                let local_n = indices.len();
                for (local, &global) in indices.iter().enumerate() {
                    local_x[local] = r[global];
                }
                self.factors[blk].solve_adjoint_in_place(&mut local_x[..local_n], hermitian);
                for (local, &global) in indices.iter().enumerate() {
                    result[global] = local_x[local];
                }
            }
        }
    }
}

// Graph partitioning helpers moved to sonium_solvers_sparse::partition.

#[cfg(debug_assertions)]
fn debug_assert_block_partition(n: usize, block_indices: &[Vec<usize>]) {
    let mut seen = vec![false; n];
    let mut count = 0usize;
    for indices in block_indices {
        for &global in indices {
            debug_assert!(global < n, "HILU block index out of bounds");
            debug_assert!(!seen[global], "HILU block indices must be disjoint");
            seen[global] = true;
            count += 1;
        }
    }
    debug_assert_eq!(count, n, "HILU block indices must cover all DOFs");
}

#[cfg(not(debug_assertions))]
fn debug_assert_block_partition(_n: usize, _block_indices: &[Vec<usize>]) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

    fn make_tridiagonal(n: usize) -> CsrMatrix<Complex64> {
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i + 1 < n {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn make_2d_laplacian(nx: usize) -> CsrMatrix<Complex64> {
        let n = nx * nx;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = Complex64::new(4.0, 0.0);
                if ix > 0 {
                    dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
                }
                if iy > 0 {
                    dense[[i, i - nx]] = Complex64::new(-1.0, 0.0);
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = Complex64::new(-1.0, 0.0);
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    #[test]
    fn checkpoint_rejects_invalid_level_ids_without_panicking() {
        let source = HiluPreconditioner::try_from_csr_uncoupled_with_config(
            &make_tridiagonal(20),
            &HiluConfig {
                min_block_size: 2,
                max_levels: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        let bytes = source.to_checkpoint_bytes().unwrap();
        assert_eq!(HILU_CHECKPOINT_FORMAT_VERSION, 1);
        // Uncoupled v1 tail: level vectors, reverse-map pairs, then None tag.
        let levels_bytes = 8 + source.levels.iter().map(|v| 8 + 8 * v.len()).sum::<usize>();
        let levels_start = bytes.len() - (8 + 16 * source.n + 1) - levels_bytes;
        let mut offset = levels_start + 8;
        let mut id_offsets = Vec::new();
        for level in &source.levels {
            offset += 8;
            for _ in level {
                id_offsets.push(offset);
                offset += 8;
            }
        }
        assert!(id_offsets.len() > 1);
        let first = id_offsets[0];
        let second_id =
            u64::from_le_bytes(bytes[id_offsets[1]..id_offsets[1] + 8].try_into().unwrap());
        for invalid in [u64::MAX, id_offsets.len() as u64, second_id] {
            let mut corrupt = bytes.clone();
            corrupt[first..first + 8].copy_from_slice(&invalid.to_le_bytes());
            let result = std::panic::catch_unwind(|| {
                HiluPreconditioner::<Complex64>::from_checkpoint_bytes(&corrupt)
            });
            assert!(result.expect("malformed level IDs must not panic").is_err());
        }
        assert!(HiluPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_ok());
    }

    #[test]
    fn checkpoint_rejects_inconsistent_uncoupled_reverse_map() {
        let source = HiluPreconditioner::try_from_csr_uncoupled_with_config(
            &make_tridiagonal(20),
            &HiluConfig {
                min_block_size: 2,
                max_levels: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(source.block_indices.len() > 1);
        let bytes = source.to_checkpoint_bytes().unwrap();
        let map_start = bytes.len() - 16 * source.n - 1;

        // Corrupt a block ID to another in-range block. Bounds-only
        // validation accepts this, but the restored map no longer describes
        // the serialized block partition.
        let original = u64::from_le_bytes(bytes[map_start..map_start + 8].try_into().unwrap());
        let replacement: u64 = if original == 0 { 1 } else { 0 };
        let mut corrupt = bytes.clone();
        corrupt[map_start..map_start + 8].copy_from_slice(&replacement.to_le_bytes());
        assert!(HiluPreconditioner::<Complex64>::from_checkpoint_bytes(&corrupt).is_err());

        // A valid block ID with a wrong, still in-range local row is also
        // rejected by the exact reverse-map check.
        let local_offset = map_start + 8;
        let local = u64::from_le_bytes(bytes[local_offset..local_offset + 8].try_into().unwrap());
        let wrong_local = (local + 1) % source.n as u64;
        let mut corrupt = bytes.clone();
        corrupt[local_offset..local_offset + 8].copy_from_slice(&wrong_local.to_le_bytes());
        assert!(HiluPreconditioner::<Complex64>::from_checkpoint_bytes(&corrupt).is_err());
        assert!(HiluPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes).is_ok());
    }

    #[test]
    fn checked_hilu_accepts_a_well_pivoted_matrix() {
        let matrix = make_tridiagonal(12);
        assert!(HiluPreconditioner::try_from_csr(&matrix).is_ok());
    }

    #[test]
    fn hilu_exposes_aggregate_setup_diagnostics() {
        let matrix = make_tridiagonal(12);
        let preconditioner = HiluPreconditioner::from_csr(&matrix);
        let diagnostics = preconditioner.diagnostics();

        assert_eq!(diagnostics.non_finite_entries, 0);
        assert!(diagnostics.min_abs_pivot.is_some());
    }

    #[test]
    fn checked_hilu_rejects_singular_block_pivots() {
        let dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(4, 4, Complex64::new(0.0, 0.0));
        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let error = match HiluPreconditioner::try_from_csr(&matrix) {
            Ok(_) => panic!("singular HILU setup must report block pivot diagnostics"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("block pivots"));
    }

    #[test]
    fn checked_hilu_rejects_invalid_configuration_and_values() {
        let matrix = make_tridiagonal(4);
        let config = HiluConfig {
            min_block_size: 0,
            ..HiluConfig::default()
        };
        let error = match HiluPreconditioner::try_from_csr_with_config(&matrix, &config) {
            Ok(_) => panic!("zero minimum block size must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("min_block_size"));

        let mut bad = make_tridiagonal(1);
        bad.values[0] = Complex64::new(f64::NAN, 0.0);
        let error = match HiluPreconditioner::try_from_csr(&bad) {
            Ok(_) => panic!("non-finite HILU input must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("NaN or infinity"));
    }

    #[test]
    fn test_hilu_apply_tridiagonal() {
        let matrix = make_tridiagonal(20);
        let config = HiluConfig {
            min_block_size: 4,
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_uncoupled_with_config(&matrix, &config);

        assert!(precond.num_blocks() > 1);
        assert!(precond.num_levels() > 1);

        let r = vec![Complex64::new(1.0, 0.0); 20];
        let x = precond.apply(&r);

        // Check finite and non-zero
        for i in 0..20 {
            assert!(x[i].norm().is_finite(), "x[{i}] is not finite");
        }
        assert!(x.iter().any(|v| v.norm() > 1e-10), "solution is all zero");
    }

    #[test]
    fn test_hilu_gmres_convergence() {
        let matrix = make_2d_laplacian(10); // 100 DOFs
        let n = matrix.num_rows;
        let b = vec![Complex64::new(1.0, 0.0); n];

        let config = GmresConfig {
            max_iterations: 200,
            restart: 50,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        // H-ILU preconditioned
        let hilu_config = HiluConfig {
            min_block_size: 16,
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_with_config(&matrix, &hilu_config);
        let sol_hilu = gmres_preconditioned(&matrix, &precond, &b, &config);

        assert!(
            sol_hilu.converged,
            "GMRES + H-ILU did not converge in {} iterations (residual={:.2e})",
            sol_hilu.iterations, sol_hilu.residual
        );

        // Verify the solution is accurate
        let residual = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&sol_hilu.x));
        let res_norm: f64 = residual.iter().map(|v| v.norm()).sum::<f64>()
            / b.iter().map(|v| v.norm()).sum::<f64>();
        assert!(
            res_norm < 1e-6,
            "Solution residual too large: {res_norm:.2e}"
        );
    }

    #[test]
    fn test_hilu_single_block_fallback() {
        // Small matrix → single block (leaf only)
        let matrix = make_tridiagonal(5);
        let config = HiluConfig {
            min_block_size: 256,
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_with_config(&matrix, &config);

        assert_eq!(precond.num_blocks(), 1);

        let r = vec![Complex64::new(1.0, 0.0); 5];
        let x = precond.apply(&r);

        // Should match sequential ILU result exactly
        let seq = crate::IluPreconditioner::from_csr(&matrix);
        let x_seq = seq.apply(&r);
        for i in 0..5 {
            assert!(
                (x[i] - x_seq[i]).norm() < 1e-12,
                "Single-block H-ILU should match sequential ILU at [{}]: hilu={:?} seq={:?}",
                i,
                x[i],
                x_seq[i]
            );
        }
    }

    #[test]
    fn test_hilu_zero_pivot_preserves_complex_phase() {
        // Regression test for issue #12: near-zero pivot fallback must preserve
        // the complex phase in H-ILU, just like in sequential ILU.
        let n = 3;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));
        dense[[0, 0]] = Complex64::new(4.0, 0.0);
        dense[[1, 1]] = Complex64::new(0.0, 1e-12); // near-zero purely imaginary
        dense[[2, 2]] = Complex64::new(5.0, 0.0);

        let matrix = CsrMatrix::from_dense(&dense, 1e-15);
        let config = HiluConfig {
            min_block_size: 256, // force single block
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_with_config(&matrix, &config);

        let b = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let result = precond.apply(&b);

        let r1 = result[1];
        assert!(
            r1.norm().is_finite(),
            "Preconditioner produced non-finite value: {:?}",
            r1
        );
        assert!(
            r1.re.abs() < 1.0,
            "Real part too large -- phase lost? re={}, im={}",
            r1.re,
            r1.im
        );
        assert!(
            r1.im.abs() > 1e10,
            "Imag part too small -- phase lost? re={}, im={}",
            r1.re,
            r1.im
        );
    }

    fn gmres_config(max_iterations: usize) -> GmresConfig<f64> {
        GmresConfig {
            max_iterations,
            restart: 50,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        }
    }

    fn gmres_iterations(
        matrix: &CsrMatrix<Complex64>,
        precond: &HiluPreconditioner<Complex64>,
        b: &[Complex64],
        max_iterations: usize,
    ) -> (usize, bool) {
        let sol = gmres_preconditioned(matrix, precond, b, &gmres_config(max_iterations));
        (sol.iterations, sol.converged)
    }

    #[test]
    fn test_hilu_default_constructor_is_coupled() {
        // The default constructors must produce the coupled (ILUM-style)
        // mode, matching the explicit coupled construction exactly.
        let matrix = make_2d_laplacian(24); // 576 DOFs
        let n = matrix.num_rows;

        let default_plain = HiluPreconditioner::from_csr(&matrix);
        assert!(
            default_plain.is_coupled(),
            "from_csr must default to the coupled mode"
        );

        let hilu_config = HiluConfig {
            min_block_size: 48,
            max_levels: None,
        };
        let default_cfg = HiluPreconditioner::from_csr_with_config(&matrix, &hilu_config);
        let explicit = HiluPreconditioner::from_csr_coupled_with_config(&matrix, &hilu_config, 0.0);
        assert!(default_cfg.is_coupled());
        assert!(explicit.is_coupled());

        // Default and explicit coupled construction must apply identically.
        let r = vec![Complex64::new(1.0, 0.0); n];
        let x_default = default_cfg.apply(&r);
        let x_explicit = explicit.apply(&r);
        for i in 0..n {
            assert!(
                (x_default[i] - x_explicit[i]).norm() < 1e-12,
                "default and coupled apply differ at [{i}]: {:?} vs {:?}",
                x_default[i],
                x_explicit[i]
            );
        }
    }

    #[test]
    fn test_hilu_coupled_single_block_fallback() {
        // No separators (single leaf) → coupled mode degenerates to plain ILU(0).
        let matrix = make_tridiagonal(5);
        let config = HiluConfig {
            min_block_size: 256,
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_coupled_with_config(&matrix, &config, 0.0);

        assert!(!precond.is_coupled());
        assert_eq!(precond.num_blocks(), 1);

        let r = vec![Complex64::new(1.0, 0.0); 5];
        let x = precond.apply(&r);
        let seq = crate::IluPreconditioner::from_csr(&matrix);
        let x_seq = seq.apply(&r);
        for i in 0..5 {
            assert!(
                (x[i] - x_seq[i]).norm() < 1e-12,
                "Coupled single-block fallback should match sequential ILU at [{}]",
                i
            );
        }
    }

    #[test]
    fn test_hilu_coupled_apply_tridiagonal() {
        // Smoke test: coupled apply produces finite, non-zero results.
        let matrix = make_tridiagonal(20);
        let config = HiluConfig {
            min_block_size: 4,
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_coupled_with_config(&matrix, &config, 0.0);

        assert!(precond.is_coupled());

        let r = vec![Complex64::new(1.0, 0.0); 20];
        let x = precond.apply(&r);
        for i in 0..20 {
            assert!(x[i].norm().is_finite(), "x[{i}] is not finite");
        }
        assert!(x.iter().any(|v| v.norm() > 1e-10), "solution is all zero");
    }

    #[test]
    fn test_hilu_coupled_gmres_convergence() {
        // 2D Laplacian: the coupled (Schur) variant must converge and be
        // accurate, like the uncoupled variant.
        let matrix = make_2d_laplacian(10); // 100 DOFs
        let n = matrix.num_rows;
        let b = vec![Complex64::new(1.0, 0.0); n];

        let hilu_config = HiluConfig {
            min_block_size: 16,
            max_levels: None,
        };
        let precond = HiluPreconditioner::from_csr_coupled_with_config(&matrix, &hilu_config, 0.0);
        assert!(precond.is_coupled());

        let config = gmres_config(200);
        let sol = gmres_preconditioned(&matrix, &precond, &b, &config);
        assert!(
            sol.converged,
            "GMRES + coupled H-ILU did not converge in {} iterations (residual={:.2e})",
            sol.iterations, sol.residual
        );

        let residual = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&sol.x));
        let res_norm: f64 = residual.iter().map(|v| v.norm()).sum::<f64>()
            / b.iter().map(|v| v.norm()).sum::<f64>();
        assert!(
            res_norm < 1e-6,
            "Solution residual too large: {res_norm:.2e}"
        );
    }

    #[test]
    fn test_hilu_coupled_fewer_iterations_than_uncoupled() {
        // The audit finding: the uncoupled mode is functionally block
        // Jacobi-ILU(0). With several blocks the coupled Schur variant must
        // need markedly fewer GMRES iterations.
        let matrix = make_2d_laplacian(24); // 576 DOFs
        let n = matrix.num_rows;
        let b = vec![Complex64::new(1.0, 0.0); n];

        let hilu_config = HiluConfig {
            min_block_size: 48,
            max_levels: None,
        };

        let uncoupled = HiluPreconditioner::from_csr_uncoupled_with_config(&matrix, &hilu_config);
        let coupled = HiluPreconditioner::from_csr_coupled_with_config(&matrix, &hilu_config, 0.0);
        assert!(coupled.is_coupled());

        let (it_u, conv_u) = gmres_iterations(&matrix, &uncoupled, &b, 500);
        let (it_c, conv_c) = gmres_iterations(&matrix, &coupled, &b, 500);
        eprintln!(
            "uncoupled: {it_u} iters (converged={conv_u}), coupled: {it_c} iters (converged={conv_c})"
        );

        assert!(
            conv_c,
            "coupled GMRES did not converge in {it_c} iterations"
        );
        assert!(
            it_c * 2 <= it_u.max(1),
            "coupled ({it_c}) should need less than half the iterations of uncoupled ({it_u})"
        );
    }

    #[test]
    fn test_hilu_coupled_iterations_scale_with_block_count() {
        // Block-Jacobi-like behaviour degrades as the number of blocks grows;
        // Schur coupling must degrade much more slowly.
        let matrix = make_2d_laplacian(32); // 1024 DOFs
        let n = matrix.num_rows;
        let b = vec![Complex64::new(1.0, 0.0); n];

        // Increasing block counts: fewer DOFs per leaf → more leaves/separators.
        let min_block_sizes = [256usize, 64, 16];
        let mut iters_uncoupled = Vec::new();
        let mut iters_coupled = Vec::new();
        for &mbs in &min_block_sizes {
            let hilu_config = HiluConfig {
                min_block_size: mbs,
                max_levels: None,
            };
            let uncoupled =
                HiluPreconditioner::from_csr_uncoupled_with_config(&matrix, &hilu_config);
            let coupled =
                HiluPreconditioner::from_csr_coupled_with_config(&matrix, &hilu_config, 0.0);
            let (it_u, _) = gmres_iterations(&matrix, &uncoupled, &b, 1000);
            let (it_c, conv_c) = gmres_iterations(&matrix, &coupled, &b, 1000);
            eprintln!(
                "min_block_size={mbs}: blocks(u)={}, blocks(c)={}, uncoupled={it_u}, coupled={it_c} (converged={conv_c})",
                uncoupled.num_blocks(),
                coupled.num_blocks()
            );
            assert!(
                conv_c,
                "coupled GMRES did not converge for min_block_size={mbs}"
            );
            iters_uncoupled.push(it_u);
            iters_coupled.push(it_c);
        }

        let growth_u = iters_uncoupled[2] as f64 / iters_uncoupled[0].max(1) as f64;
        let growth_c = iters_coupled[2] as f64 / iters_coupled[0].max(1) as f64;
        eprintln!("growth: uncoupled={growth_u:.2}x, coupled={growth_c:.2}x");

        // Coupled iteration count must grow strictly slower than uncoupled,
        // and stay low in absolute terms at the highest block count.
        assert!(
            growth_c < growth_u,
            "coupled growth ({growth_c:.2}x) should be slower than uncoupled ({growth_u:.2}x)"
        );
        assert!(
            iters_coupled[2] < iters_uncoupled[2],
            "coupled iterations ({}) should stay below uncoupled ({}) at high block count",
            iters_coupled[2],
            iters_uncoupled[2]
        );
    }
}
