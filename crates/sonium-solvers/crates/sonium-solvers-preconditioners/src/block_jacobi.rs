//! Block Jacobi Preconditioner with ILUT Subdomain Solves
//!
//! Implements domain decomposition with ZERO overlap (Block Jacobi).
//! Each subdomain gets an independent ILUT factorization.
//! All subdomain solves run in parallel when rayon is available.
//!
//! # Algorithm
//!
//! 1. Partition DOFs into `num_blocks` contiguous groups (no overlap)
//! 2. Extract local submatrix for each block
//! 3. Build `IlutPreconditioner` for each block using the provided config
//! 4. For each preconditioner apply:
//!    - Extract local RHS for each subdomain
//!    - Solve each subdomain with its local ILUT (parallel via rayon when available)
//!    - Scatter results back to global vector (no weighting needed since no overlap)

use crate::ilut::{IlutConfig, IlutPreconditioner};
use num_traits::Float;
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sparse::partition::{build_adjacency, nested_dissection};

use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Version of the portable Block-Jacobi checkpoint format.
pub const BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const BLOCK_JACOBI_CHECKPOINT_MAGIC: &[u8; 8] = b"SBJACP01";

fn block_jacobi_checkpoint_error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "Block-Jacobi checkpoint",
        reason: message.into(),
    }
}

fn block_jacobi_push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value = u64::try_from(value)
        .map_err(|_| block_jacobi_checkpoint_error("usize value does not fit checkpoint format"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

struct BlockJacobiCheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BlockJacobiCheckpointReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| block_jacobi_checkpoint_error("checkpoint offset overflow"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| block_jacobi_checkpoint_error("truncated checkpoint"))?;
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn read_u64(&mut self) -> Result<u64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(value))
    }

    fn read_len(&mut self, element_bytes: usize) -> Result<usize, SolverError> {
        let length = usize::try_from(self.read_u64()?)
            .map_err(|_| block_jacobi_checkpoint_error("checkpoint length does not fit usize"))?;
        if length > self.bytes.len().saturating_sub(self.offset) / element_bytes {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint vector length exceeds payload",
            ));
        }
        Ok(length)
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Minimum problem size where parallel subdomain solves beat sequential overhead.
/// This avoids parallelization overhead for small problems where the subdomain
/// solves are cheap. Tunable per machine via `SONIUM_BLOCK_JACOBI_PAR_SIZE`:
/// lower on fast interconnects, higher on slow ones.
const PARALLEL_SIZE_THRESHOLD: usize = 1000;

fn parallel_size_threshold() -> usize {
    std::env::var("SONIUM_BLOCK_JACOBI_PAR_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(PARALLEL_SIZE_THRESHOLD)
}

/// Strategy used to assign DOFs to blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockPartitioning {
    /// Contiguous row ranges (default; previous behavior). Cheap, but the
    /// block quality depends entirely on the matrix's DOF numbering.
    #[default]
    Contiguous,
    /// Graph-aware blocks: the matrix graph is ordered by nested dissection
    /// and the ordering is split into `num_blocks` chunks, so blocks respect
    /// the matrix connectivity (small edge cut) even when the DOF numbering
    /// is adversarial.
    NestedDissection,
}

/// Block Jacobi preconditioner with ILUT subdomain solves
///
/// Domain decomposition with zero overlap. Each subdomain is solved
/// independently using an ILUT factorization, making this embarrassingly
/// parallel.
#[derive(Debug, Clone)]
pub struct BlockJacobiPreconditioner<T: ComplexField> {
    /// Subdomain information with local ILUT factorizations
    subdomains: Vec<SubdomainIlut<T>>,
    /// Total matrix dimension
    n: usize,
    runtime: Arc<BlockJacobiRuntimeState>,
}

/// Aggregate setup diagnostics retained by the compatibility Block-Jacobi
/// constructor. Checked construction rejects these pivots before returning;
/// the historical infallible constructor keeps the factor and exposes the
/// count so callers can distinguish a regularized local solve from a clean
/// ILUT setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockJacobiDiagnostics {
    pub regularized_pivots: usize,
    pub blocks_with_regularized_pivots: usize,
    /// Non-finite values generated by local ILUT factorization.
    pub non_finite_entries: usize,
}

/// Runtime counters for checked and adjoint Block-Jacobi applications.
///
/// These counters are separate from [`BlockJacobiDiagnostics`], which
/// describes setup-time ILUT quality and remains stable across checkpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockJacobiRuntimeDiagnostics {
    pub apply_calls: u64,
    pub transpose_calls: u64,
    pub hermitian_calls: u64,
    pub failed_applications: u64,
    pub non_finite_inputs: u64,
    pub non_finite_outputs: u64,
}

#[derive(Debug, Default)]
struct BlockJacobiRuntimeState {
    apply_calls: AtomicU64,
    transpose_calls: AtomicU64,
    hermitian_calls: AtomicU64,
    failed_applications: AtomicU64,
    non_finite_inputs: AtomicU64,
    non_finite_outputs: AtomicU64,
}

impl BlockJacobiRuntimeState {
    fn snapshot(&self) -> BlockJacobiRuntimeDiagnostics {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        BlockJacobiRuntimeDiagnostics {
            apply_calls: load(&self.apply_calls),
            transpose_calls: load(&self.transpose_calls),
            hermitian_calls: load(&self.hermitian_calls),
            failed_applications: load(&self.failed_applications),
            non_finite_inputs: load(&self.non_finite_inputs),
            non_finite_outputs: load(&self.non_finite_outputs),
        }
    }
}

/// A single subdomain with its ILUT preconditioner
#[derive(Debug, Clone)]
struct SubdomainIlut<T: ComplexField> {
    /// Global indices of DOFs in this subdomain
    global_indices: Vec<usize>,
    /// Local ILUT preconditioner for this subdomain
    local_ilut: IlutPreconditioner<T>,
}

impl<T: ComplexField> BlockJacobiPreconditioner<T> {
    /// Create a Block Jacobi preconditioner from a CSR matrix
    ///
    /// Partitions rows into `num_blocks` contiguous blocks with no overlap,
    /// then builds an ILUT factorization for each block.
    ///
    /// # Arguments
    /// * `matrix` - The global sparse matrix
    /// * `num_blocks` - Number of blocks (subdomains)
    /// * `ilut_config` - Configuration for each subdomain's ILUT factorization
    pub fn from_csr(matrix: &CsrMatrix<T>, num_blocks: usize, ilut_config: IlutConfig) -> Self {
        Self::from_csr_with_partitioning_unchecked(
            matrix,
            num_blocks,
            ilut_config,
            BlockPartitioning::Contiguous,
        )
    }

    /// Fallible Block-Jacobi construction that validates the square CSR
    /// storage before any graph partition or local extraction.
    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
        num_blocks: usize,
        ilut_config: IlutConfig,
    ) -> Result<Self, SolverError> {
        Self::try_from_csr_with_partitioning(
            matrix,
            num_blocks,
            ilut_config,
            BlockPartitioning::Contiguous,
        )
    }

    /// Create a Block Jacobi preconditioner with an explicit partitioning
    /// strategy. With [`BlockPartitioning::Contiguous`] this is identical to
    /// [`Self::from_csr`].
    pub fn from_csr_with_partitioning(
        matrix: &CsrMatrix<T>,
        num_blocks: usize,
        ilut_config: IlutConfig,
        partitioning: BlockPartitioning,
    ) -> Self {
        Self::from_csr_with_partitioning_unchecked(matrix, num_blocks, ilut_config, partitioning)
    }

    pub fn try_from_csr_with_partitioning(
        matrix: &CsrMatrix<T>,
        num_blocks: usize,
        ilut_config: IlutConfig,
        partitioning: BlockPartitioning,
    ) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        if matrix.num_rows != matrix.num_cols {
            return Err(SolverError::OperationFailed {
                operation: "block Jacobi construction",
                reason: format!(
                    "matrix must be square, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        Self::try_build_from_csr_with_partitioning(matrix, num_blocks, ilut_config, partitioning)
    }

    fn from_csr_with_partitioning_unchecked(
        matrix: &CsrMatrix<T>,
        num_blocks: usize,
        ilut_config: IlutConfig,
        partitioning: BlockPartitioning,
    ) -> Self {
        let n = matrix.num_rows;
        if n == 0 {
            return Self {
                subdomains: Vec::new(),
                n,
                runtime: Arc::new(BlockJacobiRuntimeState::default()),
            };
        }
        let num_blocks = num_blocks.max(1).min(n);
        let blocks: Vec<Vec<usize>> = match partitioning {
            BlockPartitioning::Contiguous => contiguous_blocks(n, num_blocks),
            BlockPartitioning::NestedDissection => nested_dissection_blocks(matrix, num_blocks),
        };
        let subdomains: Vec<SubdomainIlut<T>> = blocks
            .into_par_iter()
            .map(|global_indices| {
                let local_matrix = extract_submatrix(matrix, &global_indices);
                let local_ilut =
                    IlutPreconditioner::from_csr_with_config(&local_matrix, &ilut_config);
                SubdomainIlut {
                    global_indices,
                    local_ilut,
                }
            })
            .collect();
        Self {
            subdomains,
            n,
            runtime: Arc::new(BlockJacobiRuntimeState::default()),
        }
    }

    fn try_build_from_csr_with_partitioning(
        matrix: &CsrMatrix<T>,
        num_blocks: usize,
        ilut_config: IlutConfig,
        partitioning: BlockPartitioning,
    ) -> Result<Self, SolverError> {
        let n = matrix.num_rows;
        if n == 0 {
            return Ok(Self {
                subdomains: Vec::new(),
                n,
                runtime: Arc::new(BlockJacobiRuntimeState::default()),
            });
        }
        let num_blocks = num_blocks.max(1).min(n);

        // Step 1: Partition DOFs into blocks (no overlap)
        let blocks: Vec<Vec<usize>> = match partitioning {
            BlockPartitioning::Contiguous => contiguous_blocks(n, num_blocks),
            BlockPartitioning::NestedDissection => nested_dissection_blocks(matrix, num_blocks),
        };

        let subdomains: Vec<SubdomainIlut<T>> = blocks
            .into_par_iter()
            .map(|global_indices| -> Result<SubdomainIlut<T>, SolverError> {
                // Step 2: Extract local submatrix
                let local_matrix = extract_submatrix(matrix, &global_indices);

                // Step 3: Build ILUT preconditioner for this block
                let local_ilut =
                    IlutPreconditioner::try_from_csr_with_config(&local_matrix, &ilut_config)?;

                Ok(SubdomainIlut {
                    global_indices,
                    local_ilut,
                })
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            subdomains,
            n,
            runtime: Arc::new(BlockJacobiRuntimeState::default()),
        })
    }

    /// Create with automatic block count based on available threads.
    ///
    /// Two caps apply:
    /// * `MIN_DOFS_PER_BLOCK` — each block holds at least this many DOFs;
    ///   below it the ILUT factorisation's fixed overhead per block
    ///   dominates the actual work.
    /// * `MAX_USEFUL_BLOCKS` — past this many partitions, total ILUT work
    ///   (which grows roughly with block count due to lost cross-block
    ///   coupling) plus simultaneous-factorisation memory pressure makes
    ///   setup *slower* on many-core x86 boxes. Empirically the plateau
    ///   sits around 32 blocks regardless of how many threads are
    ///   available, because ILUT setup is bandwidth-bound and saturates
    ///   long before all P-cores can do useful work.
    pub fn from_csr_auto(matrix: &CsrMatrix<T>, ilut_config: IlutConfig) -> Self {
        const MIN_DOFS_PER_BLOCK: usize = 50_000;
        const MAX_USEFUL_BLOCKS: usize = 32;
        let num_threads = rayon::current_num_threads();
        let by_size = matrix.num_rows.div_ceil(MIN_DOFS_PER_BLOCK).max(1);
        let num_blocks = num_threads.min(by_size).min(MAX_USEFUL_BLOCKS);
        Self::from_csr(matrix, num_blocks, ilut_config)
    }

    /// Checked automatic constructor using the same bounded block-count
    /// policy as [`Self::from_csr_auto`].
    pub fn try_from_csr_auto(
        matrix: &CsrMatrix<T>,
        ilut_config: IlutConfig,
    ) -> Result<Self, SolverError> {
        const MIN_DOFS_PER_BLOCK: usize = 50_000;
        const MAX_USEFUL_BLOCKS: usize = 32;
        let num_threads = rayon::current_num_threads();
        let by_size = matrix.num_rows.div_ceil(MIN_DOFS_PER_BLOCK).max(1);
        let num_blocks = num_threads.min(by_size).min(MAX_USEFUL_BLOCKS);
        Self::try_from_csr(matrix, num_blocks, ilut_config)
    }

    /// Get statistics about the block decomposition
    ///
    /// Returns `(num_blocks, min_size, max_size)`
    pub fn stats(&self) -> (usize, usize, usize) {
        let num_blocks = self.subdomains.len();
        let min_size = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .min()
            .unwrap_or(0);
        let max_size = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .max()
            .unwrap_or(0);
        (num_blocks, min_size, max_size)
    }

    /// Return aggregate diagnostics from all local ILUT factorizations.
    pub fn diagnostics(&self) -> BlockJacobiDiagnostics {
        let mut diagnostics = BlockJacobiDiagnostics {
            regularized_pivots: 0,
            blocks_with_regularized_pivots: 0,
            non_finite_entries: 0,
        };
        for subdomain in &self.subdomains {
            let count = subdomain.local_ilut.regularized_pivots().len();
            diagnostics.regularized_pivots += count;
            diagnostics.blocks_with_regularized_pivots += usize::from(count != 0);
            diagnostics.non_finite_entries += subdomain.local_ilut.non_finite_entries();
        }
        diagnostics
    }

    /// Snapshot runtime application counters.
    pub fn runtime_diagnostics(&self) -> BlockJacobiRuntimeDiagnostics {
        self.runtime.snapshot()
    }

    /// Serialize the block partition and every nested ILUT factor.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let mut seen = vec![false; self.n];
        let mut nested_checkpoints = Vec::with_capacity(self.subdomains.len());

        for subdomain in &self.subdomains {
            if subdomain.global_indices.is_empty() {
                return Err(block_jacobi_checkpoint_error(
                    "checkpoint cannot contain an empty block",
                ));
            }
            if subdomain.local_ilut.dimension() != subdomain.global_indices.len() {
                return Err(block_jacobi_checkpoint_error(
                    "block indices and ILUT dimension differ",
                ));
            }
            for &global_index in &subdomain.global_indices {
                let slot = seen.get_mut(global_index).ok_or_else(|| {
                    block_jacobi_checkpoint_error("block index is outside matrix dimension")
                })?;
                if *slot {
                    return Err(block_jacobi_checkpoint_error(
                        "checkpoint contains duplicate global index",
                    ));
                }
                *slot = true;
            }
            nested_checkpoints.push(subdomain.local_ilut.to_checkpoint_bytes()?);
        }

        if seen.iter().any(|present| !present) {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint block partition does not cover all DOFs",
            ));
        }

        let mut output = Vec::new();
        output.extend_from_slice(BLOCK_JACOBI_CHECKPOINT_MAGIC);
        output.extend_from_slice(&BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        block_jacobi_push_u64(&mut output, self.n)?;
        block_jacobi_push_u64(&mut output, self.subdomains.len())?;
        for (subdomain, nested_checkpoint) in self.subdomains.iter().zip(nested_checkpoints) {
            block_jacobi_push_u64(&mut output, subdomain.global_indices.len())?;
            for &global_index in &subdomain.global_indices {
                block_jacobi_push_u64(&mut output, global_index)?;
            }
            block_jacobi_push_u64(&mut output, nested_checkpoint.len())?;
            output.extend_from_slice(&nested_checkpoint);
        }
        Ok(output)
    }

    /// Restore a block-Jacobi preconditioner from [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = BlockJacobiCheckpointReader::new(bytes);
        if reader.take(8)? != BLOCK_JACOBI_CHECKPOINT_MAGIC {
            return Err(block_jacobi_checkpoint_error("invalid checkpoint magic"));
        }
        if reader.read_u16()? != BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION {
            return Err(block_jacobi_checkpoint_error(
                "unsupported checkpoint version",
            ));
        }
        if reader.read_u8()? as usize != std::mem::size_of::<T>() {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint scalar size mismatch",
            ));
        }
        if reader.read_u8()? as usize != std::mem::size_of::<T::Real>() {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint real scalar size mismatch",
            ));
        }

        let n = reader.read_len(1)?;
        let block_count = reader.read_len(16)?;
        if (n == 0 && block_count != 0) || (n != 0 && block_count == 0) {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint block count does not match dimension",
            ));
        }
        if block_count > n {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint has more blocks than DOFs",
            ));
        }

        let mut seen = vec![false; n];
        let mut subdomains = Vec::with_capacity(block_count);
        let mut covered = 0_usize;
        for _ in 0..block_count {
            let index_count = reader.read_len(8)?;
            if index_count == 0 {
                return Err(block_jacobi_checkpoint_error(
                    "checkpoint contains an empty block",
                ));
            }
            covered = covered
                .checked_add(index_count)
                .ok_or_else(|| block_jacobi_checkpoint_error("block index count overflow"))?;
            if covered > n {
                return Err(block_jacobi_checkpoint_error(
                    "checkpoint blocks contain too many DOFs",
                ));
            }
            let mut global_indices = Vec::with_capacity(index_count);
            for _ in 0..index_count {
                let global_index = usize::try_from(reader.read_u64()?).map_err(|_| {
                    block_jacobi_checkpoint_error("global index does not fit usize")
                })?;
                let slot = seen.get_mut(global_index).ok_or_else(|| {
                    block_jacobi_checkpoint_error("block index is outside matrix dimension")
                })?;
                if *slot {
                    return Err(block_jacobi_checkpoint_error(
                        "checkpoint contains duplicate global index",
                    ));
                }
                *slot = true;
                global_indices.push(global_index);
            }

            let nested_len = reader.read_len(1)?;
            if nested_len == 0 {
                return Err(block_jacobi_checkpoint_error(
                    "checkpoint contains an empty ILUT payload",
                ));
            }
            let nested_bytes = reader.take(nested_len)?;
            let local_ilut = IlutPreconditioner::from_checkpoint_bytes(nested_bytes)?;
            if local_ilut.dimension() != index_count {
                return Err(block_jacobi_checkpoint_error(
                    "block indices and ILUT dimension differ",
                ));
            }
            subdomains.push(SubdomainIlut {
                global_indices,
                local_ilut,
            });
        }

        if covered != n || seen.iter().any(|present| !present) {
            return Err(block_jacobi_checkpoint_error(
                "checkpoint block partition does not cover all DOFs",
            ));
        }
        if !reader.is_at_end() {
            return Err(block_jacobi_checkpoint_error("trailing bytes"));
        }
        Ok(Self {
            subdomains,
            n,
            runtime: Arc::new(BlockJacobiRuntimeState::default()),
        })
    }
}

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

/// Contiguous row-range blocks (balanced sizes, `n % num_blocks` larger
/// blocks first).
fn contiguous_blocks(n: usize, num_blocks: usize) -> Vec<Vec<usize>> {
    if n == 0 || num_blocks == 0 {
        return Vec::new();
    }
    let base_size = n / num_blocks;
    let remainder = n % num_blocks;

    let mut start = 0;
    (0..num_blocks)
        .map(|i| {
            let size = base_size + if i < remainder { 1 } else { 0 };
            let block: Vec<usize> = (start..start + size).collect();
            start += size;
            block
        })
        .collect()
}

/// Graph-aware blocks via nested dissection.
///
/// The dissection tree is post-ordered by construction (left subtree, right
/// subtree, separator), so concatenating the node DOF lists yields a
/// fill-reducing ordering that clusters connected DOFs. Splitting that
/// ordering into `num_blocks` contiguous chunks gives exactly `num_blocks`
/// blocks with a small edge cut, independent of the original DOF numbering.
fn nested_dissection_blocks<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    num_blocks: usize,
) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;
    if n == 0 || num_blocks == 0 {
        return Vec::new();
    }
    let adjacency = build_adjacency(matrix);
    let all_indices: Vec<usize> = (0..n).collect();

    // Deep dissection: small leaves give a fine-grained ordering; the block
    // count is fixed afterwards by chunking, so this only controls ordering
    // quality, not the number of blocks.
    let mut nodes = Vec::new();
    nested_dissection(&adjacency, &all_indices, 8, 64, 0, &mut nodes);

    let perm: Vec<usize> = nodes
        .iter()
        .flat_map(|nd| nd.global_indices.iter().copied())
        .collect();
    debug_assert_eq!(perm.len(), n, "dissection must cover every DOF once");

    // Chunk the ordering like contiguous_blocks does for the identity order.
    let base_size = n / num_blocks;
    let remainder = n % num_blocks;
    let mut start = 0;
    (0..num_blocks)
        .map(|i| {
            let size = base_size + if i < remainder { 1 } else { 0 };
            let block = perm[start..start + size].to_vec();
            start += size;
            block
        })
        .collect()
}

/// Extract a submatrix corresponding to the given global indices
///
/// Builds a local CSR matrix containing only rows and columns that
/// appear in `global_indices`.
fn extract_submatrix<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    global_indices: &[usize],
) -> CsrMatrix<T> {
    let local_n = global_indices.len();

    // Build global-to-local index mapping
    let mut global_to_local = vec![usize::MAX; matrix.num_rows];
    for (local_idx, &global_idx) in global_indices.iter().enumerate() {
        global_to_local[global_idx] = local_idx;
    }

    // Extract local matrix in CSR format
    let mut local_values = Vec::new();
    let mut local_col_indices = Vec::new();
    let mut local_row_ptrs = vec![0];

    for &global_row in global_indices {
        for idx in matrix.row_ptrs[global_row]..matrix.row_ptrs[global_row + 1] {
            let global_col = matrix.col_indices[idx];
            let local_col = global_to_local[global_col];
            if local_col != usize::MAX {
                local_values.push(matrix.values[idx]);
                local_col_indices.push(local_col);
            }
        }
        local_row_ptrs.push(local_values.len());
    }

    CsrMatrix::from_raw_parts(
        local_n,
        local_n,
        local_row_ptrs,
        local_col_indices,
        local_values,
    )
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for BlockJacobiPreconditioner<T> {
    fn try_apply(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if r.len() != self.n {
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::DimensionMismatch {
                expected: self.n,
                got: r.len(),
            });
        }
        if r.iter().any(|value| !value.norm().is_finite()) {
            self.runtime
                .non_finite_inputs
                .fetch_add(1, Ordering::Relaxed);
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::OperationFailed {
                operation: "Block-Jacobi apply",
                reason: "input contains NaN or infinity".into(),
            });
        }
        let applied = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.apply(r)))
        {
            Ok(applied) => applied,
            Err(_) => {
                self.runtime
                    .failed_applications
                    .fetch_add(1, Ordering::Relaxed);
                return Err(SolverError::OperationFailed {
                    operation: "Block-Jacobi apply",
                    reason: "preconditioner application panicked".into(),
                });
            }
        };
        if applied.len() != self.n {
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::DimensionMismatch {
                expected: self.n,
                got: applied.len(),
            });
        }
        if applied.iter().any(|value| !value.norm().is_finite()) {
            self.runtime
                .non_finite_outputs
                .fetch_add(1, Ordering::Relaxed);
            self.runtime
                .failed_applications
                .fetch_add(1, Ordering::Relaxed);
            return Err(SolverError::OperationFailed {
                operation: "Block-Jacobi apply",
                reason: "output contains NaN or infinity".into(),
            });
        }
        Ok(applied)
    }

    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        self.runtime.apply_calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(r.len(), self.n, "BlockJacobi RHS length mismatch");
        assert_eq!(y.len(), self.n, "BlockJacobi output length mismatch");

        if self.n >= parallel_size_threshold() && self.subdomains.len() > 1 {
            self.apply_parallel_into(r, y);
        } else {
            self.apply_sequential_into(r, y);
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_transpose_into(r, &mut y);
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_transpose_into(&self, r: &[T], y: &mut [T]) {
        self.runtime.transpose_calls.fetch_add(1, Ordering::Relaxed);
        self.apply_adjoint_into(r, y, false);
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        self.apply_hermitian_into(r, &mut y);
        y
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn apply_hermitian_into(&self, r: &[T], y: &mut [T]) {
        self.runtime.hermitian_calls.fetch_add(1, Ordering::Relaxed);
        self.apply_adjoint_into(r, y, true);
    }
}

impl<T: ComplexField + Send + Sync> BlockJacobiPreconditioner<T> {
    fn apply_sequential_into(&self, r: &[T], y: &mut [T]) {
        y.fill(T::zero());

        let max_local_n = self
            .subdomains
            .iter()
            .map(|s| s.global_indices.len())
            .max()
            .unwrap_or(0);
        let mut local_work = vec![T::zero(); max_local_n];

        for subdomain in &self.subdomains {
            let local_n = subdomain.global_indices.len();

            // Extract local RHS
            for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                local_work[local_idx] = r[global_idx];
            }

            subdomain
                .local_ilut
                .solve_in_place(&mut local_work[..local_n]);

            // Scatter back to global (no weighting needed since no overlap)
            for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                y[global_idx] = local_work[local_idx];
            }
        }
    }

    fn apply_parallel_into(&self, r: &[T], y: &mut [T]) {
        let r_slice = r;
        y.fill(T::zero());
        // Debug-only disjointness check: subdomain index sets must be
        // pairwise non-overlapping and in bounds, else the scatter below
        // is a data race.
        debug_assert!({
            let mut seen = vec![false; y.len()];
            self.subdomains.iter().all(|sd| {
                sd.global_indices.iter().all(|&g| {
                    if g >= seen.len() || seen[g] {
                        return false;
                    }
                    seen[g] = true;
                    true
                })
            })
        });
        let result_ptr = DisjointScatterPtr(y.as_mut_ptr());

        self.subdomains
            .par_iter()
            .for_each_init(Vec::<T>::new, |local_work, subdomain| {
                let local_n = subdomain.global_indices.len();
                local_work.resize(local_n, T::zero());
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    local_work[local_idx] = r_slice[global_idx];
                }

                subdomain.local_ilut.solve_in_place(local_work);

                // Block Jacobi partitions are non-overlapping, so each global
                // result slot is written by exactly one subdomain.
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    unsafe {
                        result_ptr.write(global_idx, local_work[local_idx]);
                    }
                }
            });
    }

    fn apply_adjoint_into(&self, r: &[T], y: &mut [T], hermitian: bool) {
        assert_eq!(r.len(), self.n, "BlockJacobi RHS length mismatch");
        assert_eq!(y.len(), self.n, "BlockJacobi output length mismatch");

        if self.n >= parallel_size_threshold() && self.subdomains.len() > 1 {
            let r_slice = r;
            y.fill(T::zero());
            // Debug-only disjointness check: subdomain index sets must be
            // pairwise non-overlapping and in bounds, else the scatter below
            // is a data race.
            debug_assert!({
                let mut seen = vec![false; y.len()];
                self.subdomains.iter().all(|sd| {
                    sd.global_indices.iter().all(|&g| {
                        if g >= seen.len() || seen[g] {
                            return false;
                        }
                        seen[g] = true;
                        true
                    })
                })
            });
            let result_ptr = DisjointScatterPtr(y.as_mut_ptr());

            self.subdomains.par_iter().for_each_init(
                || (Vec::<T>::new(), Vec::<T>::new()),
                |(local_work, local_output), subdomain| {
                    let local_n = subdomain.global_indices.len();
                    local_work.resize(local_n, T::zero());
                    local_output.resize(local_n, T::zero());
                    for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                        local_work[local_idx] = r_slice[global_idx];
                    }

                    if hermitian {
                        subdomain
                            .local_ilut
                            .apply_hermitian_into(local_work, local_output);
                    } else {
                        subdomain
                            .local_ilut
                            .apply_transpose_into(local_work, local_output);
                    }

                    for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                        unsafe {
                            result_ptr.write(global_idx, local_output[local_idx]);
                        }
                    }
                },
            );
        } else {
            y.fill(T::zero());
            let max_local_n = self
                .subdomains
                .iter()
                .map(|subdomain| subdomain.global_indices.len())
                .max()
                .unwrap_or(0);
            let mut local_work = vec![T::zero(); max_local_n];
            let mut local_output = vec![T::zero(); max_local_n];

            for subdomain in &self.subdomains {
                let local_n = subdomain.global_indices.len();
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    local_work[local_idx] = r[global_idx];
                }
                if hermitian {
                    subdomain
                        .local_ilut
                        .apply_hermitian_into(&local_work[..local_n], &mut local_output[..local_n]);
                } else {
                    subdomain
                        .local_ilut
                        .apply_transpose_into(&local_work[..local_n], &mut local_output[..local_n]);
                }
                for (local_idx, &global_idx) in subdomain.global_indices.iter().enumerate() {
                    y[global_idx] = local_output[local_idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

    fn create_test_matrix() -> CsrMatrix<Complex64> {
        let n = 20;
        let mut dense =
            sonium_solvers_core::dense::DenseMatrix::from_elem(n, n, Complex64::new(0.0, 0.0));

        // Create a diagonally dominant matrix
        for i in 0..n {
            dense[[i, i]] = Complex64::new(4.0, 0.0);
            if i > 0 {
                dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
            }
            if i < n - 1 {
                dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
            }
            if i >= 5 {
                dense[[i, i - 5]] = Complex64::new(-0.5, 0.0);
            }
            if i < n - 5 {
                dense[[i, i + 5]] = Complex64::new(-0.5, 0.0);
            }
        }

        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn create_diagonal_matrix(n: usize, value: Complex64) -> CsrMatrix<Complex64> {
        let triplets = (0..n).map(|i| (i, i, value)).collect();
        CsrMatrix::from_triplets(n, n, triplets)
    }

    fn create_pivoted_matrix() -> CsrMatrix<Complex64> {
        let dense = sonium_solvers_core::array![
            [
                Complex64::new(1e-8, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1e-8, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0)
            ],
            [
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1e-8, 0.0)
            ],
        ];
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn assert_vectors_close(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (*a - *e).norm())
            .fold(0.0, f64::max);
        assert!(
            max_error < 1e-10,
            "maximum vector error {max_error:e} exceeds pivoted ILUT reference"
        );
    }

    #[test]
    fn test_block_jacobi_basic_apply() {
        let matrix = create_test_matrix();
        let config = IlutConfig::default();
        let precond = BlockJacobiPreconditioner::from_csr(&matrix, 4, config);

        let r = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&r);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|x| x.norm() < 100.0));
    }

    #[test]
    fn test_block_jacobi_with_gmres() {
        let matrix = create_test_matrix();
        let config = IlutConfig::default();
        let precond = BlockJacobiPreconditioner::from_csr(&matrix, 4, config);

        let b = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let gmres_config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_preconditioned(&matrix, &precond, &b, &gmres_config);
        assert!(
            sol.converged,
            "GMRES with Block Jacobi should converge, iterations: {}, residual: {}",
            sol.iterations, sol.residual,
        );
    }

    #[test]
    fn test_block_jacobi_stats() {
        let matrix = create_test_matrix();
        let config = IlutConfig::default();
        let precond = BlockJacobiPreconditioner::from_csr(&matrix, 4, config);

        let (num_blocks, min_size, max_size) = precond.stats();

        assert_eq!(num_blocks, 4);
        assert!(min_size > 0);
        assert!(max_size >= min_size);
        // 20 rows / 4 blocks = 5 per block
        assert_eq!(min_size, 5);
        assert_eq!(max_size, 5);
    }

    #[test]
    fn test_block_jacobi_parallel_apply_matches_diagonal_inverse() {
        let n = PARALLEL_SIZE_THRESHOLD + 64;
        let matrix = create_diagonal_matrix(n, Complex64::new(2.0, 0.0));
        let precond = BlockJacobiPreconditioner::from_csr(&matrix, 8, IlutConfig::default());

        let r = (0..n)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.0))
            .collect::<Vec<_>>();
        let result = precond.apply(&r);

        assert_eq!(result.len(), n);
        for (actual, rhs) in result.iter().zip(&r) {
            assert!((*actual - *rhs / Complex64::new(2.0, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn test_block_jacobi_single_block() {
        // Single block should be equivalent to a full ILUT on the whole matrix
        let matrix = create_test_matrix();
        let config = IlutConfig::default();
        let precond = BlockJacobiPreconditioner::from_csr(&matrix, 1, config);

        let (num_blocks, _, _) = precond.stats();
        assert_eq!(num_blocks, 1, "Should have exactly one block");

        let b = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let gmres_config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_preconditioned(&matrix, &precond, &b, &gmres_config);
        assert!(
            sol.converged,
            "Single-block Block Jacobi should converge, residual: {}",
            sol.residual,
        );
    }

    #[test]
    fn block_jacobi_preserves_pivoted_ilut_forward_and_adjoint_actions() {
        let matrix = create_pivoted_matrix();
        let config = IlutConfig {
            drop_tolerance: 0.0,
            fill_factor: 100,
            pivot_tolerance: 1e-15,
            pivoting: true,
        };
        let reference = IlutPreconditioner::from_csr_with_config(&matrix, &config);
        let block = BlockJacobiPreconditioner::from_csr(&matrix, 1, config);
        let rhs = vec![
            Complex64::new(0.7, -0.2),
            Complex64::new(-1.1, 0.4),
            Complex64::new(0.3, 0.9),
            Complex64::new(1.4, -0.6),
            Complex64::new(-0.5, 0.8),
        ];

        let expected_forward = reference.apply(&rhs);
        let expected_transpose = reference.apply_transpose(&rhs);
        let mut raw_adjoint = rhs.clone();
        reference.solve_adjoint_in_place(&mut raw_adjoint, false);
        assert!(
            raw_adjoint
                .iter()
                .zip(&expected_transpose)
                .any(|(raw, expected)| (*raw - *expected).norm() > 1e-8),
            "fixture must distinguish raw adjoint solve from ILUTP application"
        );

        assert_vectors_close(&block.apply(&rhs), &expected_forward);
        assert_vectors_close(&block.apply_transpose(&rhs), &expected_transpose);
        assert_vectors_close(
            &block.apply_hermitian(&rhs),
            &reference.apply_hermitian(&rhs),
        );
    }

    #[test]
    fn test_block_jacobi_many_blocks() {
        // More blocks than rows: should clamp to n blocks (row-by-row)
        let matrix = create_test_matrix();
        let config = IlutConfig::default();
        let precond = BlockJacobiPreconditioner::from_csr(&matrix, 100, config);

        let (num_blocks, min_size, max_size) = precond.stats();
        // Clamped to n=20 blocks
        assert_eq!(num_blocks, 20, "Should clamp to n blocks");
        assert_eq!(min_size, 1, "Each block should have exactly 1 row");
        assert_eq!(max_size, 1, "Each block should have exactly 1 row");

        let b = (0..20)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let gmres_config = GmresConfig {
            max_iterations: 100,
            restart: 20,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol = gmres_preconditioned(&matrix, &precond, &b, &gmres_config);
        assert!(
            sol.converged,
            "Many-blocks Block Jacobi should converge, residual: {}",
            sol.residual,
        );
    }

    /// 5-point Laplacian on an m×m grid with a scrambled (fixed-seed) DOF
    /// numbering: contiguous row ranges then mix DOFs from all over the
    /// grid, while graph-aware partitioning can recover the locality.
    fn permuted_laplacian_2d(m: usize) -> CsrMatrix<Complex64> {
        let n = m * m;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut state: u64 = 0x9E3779B97F4A7C15;
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((state >> 33) as usize) % (i + 1);
            perm.swap(i, j);
        }

        // A'[perm[i]][perm[j]] = A[i][j]
        let mut triplets = Vec::new();
        let mut push = |r: usize, c: usize, v: f64| {
            triplets.push((perm[r], perm[c], Complex64::new(v, 0.0)));
        };
        for r in 0..m {
            for c in 0..m {
                let i = r * m + c;
                push(i, i, 4.0);
                if r > 0 {
                    push(i, i - m, -1.0);
                }
                if r + 1 < m {
                    push(i, i + m, -1.0);
                }
                if c > 0 {
                    push(i, i - 1, -1.0);
                }
                if c + 1 < m {
                    push(i, i + 1, -1.0);
                }
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    #[test]
    fn test_block_jacobi_nested_dissection_partitioning() {
        // On a scrambled 2D Laplacian, contiguous blocks are essentially
        // random DOF sets (large edge cut, near-diagonal local problems),
        // while nested-dissection blocks recover grid locality. The
        // graph-aware partition must need fewer GMRES iterations.
        //
        // Measured on a 20x20 grid (n = 400), 8 blocks, tol 1e-8:
        //   contiguous:        80 iterations
        //   nested dissection: 48 iterations
        let m = 20;
        let n = m * m;
        let num_blocks = 8;
        let matrix = permuted_laplacian_2d(m);

        let contiguous =
            BlockJacobiPreconditioner::from_csr(&matrix, num_blocks, IlutConfig::default());
        let graph = BlockJacobiPreconditioner::from_csr_with_partitioning(
            &matrix,
            num_blocks,
            IlutConfig::default(),
            BlockPartitioning::NestedDissection,
        );

        // The graph-aware blocks must form a genuine partition of 0..n.
        let (nb, min_size, max_size) = graph.stats();
        assert_eq!(nb, num_blocks);
        assert!(min_size > 0 && max_size >= min_size);
        let mut covered: Vec<usize> = graph
            .subdomains
            .iter()
            .flat_map(|s| s.global_indices.iter().copied())
            .collect();
        covered.sort_unstable();
        assert_eq!(
            covered,
            (0..n).collect::<Vec<_>>(),
            "blocks must partition the DOFs"
        );

        let b: Vec<_> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), 0.0))
            .collect();
        let gmres_config = GmresConfig {
            max_iterations: 500,
            restart: 40,
            tolerance: 1e-8,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        let sol_contiguous = gmres_preconditioned(&matrix, &contiguous, &b, &gmres_config);
        let sol_graph = gmres_preconditioned(&matrix, &graph, &b, &gmres_config);
        eprintln!(
            "Block Jacobi on scrambled 2D Laplacian (n={n}, {num_blocks} blocks): \
             contiguous={} its (converged={}), nested-dissection={} its (converged={})",
            sol_contiguous.iterations,
            sol_contiguous.converged,
            sol_graph.iterations,
            sol_graph.converged,
        );

        assert!(
            sol_graph.converged,
            "graph-partitioned Block Jacobi should converge, residual: {}",
            sol_graph.residual,
        );
        assert!(
            sol_graph.iterations < sol_contiguous.iterations,
            "graph partitioning should beat contiguous: {} vs {} iterations",
            sol_graph.iterations,
            sol_contiguous.iterations,
        );
    }

    #[test]
    fn test_block_jacobi_contiguous_is_default() {
        // from_csr must remain the contiguous partitioning (backward compat).
        let matrix = permuted_laplacian_2d(6);
        let default = BlockJacobiPreconditioner::from_csr(&matrix, 3, IlutConfig::default());
        let explicit = BlockJacobiPreconditioner::from_csr_with_partitioning(
            &matrix,
            3,
            IlutConfig::default(),
            BlockPartitioning::Contiguous,
        );
        for (a, b) in default.subdomains.iter().zip(explicit.subdomains.iter()) {
            assert_eq!(a.global_indices, b.global_indices);
        }
    }

    #[test]
    fn block_jacobi_exposes_aggregate_setup_diagnostics() {
        let matrix = create_test_matrix();
        let preconditioner = BlockJacobiPreconditioner::from_csr(&matrix, 4, IlutConfig::default());
        let diagnostics = preconditioner.diagnostics();

        assert_eq!(diagnostics.regularized_pivots, 0);
        assert_eq!(diagnostics.blocks_with_regularized_pivots, 0);
        assert_eq!(diagnostics.non_finite_entries, 0);
    }

    #[test]
    fn block_jacobi_aggregates_generated_non_finite_local_factors() {
        let matrix = CsrMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1e-100_f64, 1e154, 1e154, 1e100],
        );
        let config = IlutConfig {
            drop_tolerance: 0.0,
            pivot_tolerance: 1e-300,
            pivoting: false,
            ..IlutConfig::default()
        };

        let compatibility = BlockJacobiPreconditioner::from_csr(&matrix, 1, config.clone());
        assert!(compatibility.diagnostics().non_finite_entries > 0);

        let error = BlockJacobiPreconditioner::try_from_csr(&matrix, 1, config)
            .expect_err("checked Block-Jacobi must reject generated local factors");
        assert!(matches!(error, SolverError::OperationFailed { .. }));
        assert!(error.to_string().contains("generated non-finite factor"));
    }

    #[test]
    fn block_jacobi_checkpoint_preserves_partition_metadata() {
        let matrix = create_test_matrix();
        let source = BlockJacobiPreconditioner::from_csr_with_partitioning(
            &matrix,
            4,
            IlutConfig::default(),
            BlockPartitioning::NestedDissection,
        );
        let bytes = source
            .to_checkpoint_bytes()
            .expect("serialize block Jacobi");
        let restored = BlockJacobiPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
            .expect("restore block Jacobi");
        let source_partition: Vec<&[usize]> = source
            .subdomains
            .iter()
            .map(|subdomain| subdomain.global_indices.as_slice())
            .collect();
        let restored_partition: Vec<&[usize]> = restored
            .subdomains
            .iter()
            .map(|subdomain| subdomain.global_indices.as_slice())
            .collect();
        assert_eq!(source_partition, restored_partition);
    }

    #[test]
    fn block_jacobi_empty_matrix_is_an_empty_preconditioner() {
        let matrix = CsrMatrix::from_triplets(0, 0, Vec::<(usize, usize, Complex64)>::new());

        for partitioning in [
            BlockPartitioning::Contiguous,
            BlockPartitioning::NestedDissection,
        ] {
            let legacy = BlockJacobiPreconditioner::from_csr_with_partitioning(
                &matrix,
                0,
                IlutConfig::default(),
                partitioning,
            );
            assert_eq!(legacy.stats(), (0, 0, 0));
            assert!(legacy.apply(&[]).is_empty());

            let checked = BlockJacobiPreconditioner::try_from_csr_with_partitioning(
                &matrix,
                0,
                IlutConfig::default(),
                partitioning,
            )
            .expect("an empty square matrix is a valid empty preconditioner");
            assert_eq!(checked.stats(), (0, 0, 0));
            assert!(checked.apply(&[]).is_empty());
        }
    }
}
