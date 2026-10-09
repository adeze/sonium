//! NUMA-aware partitioned CSR matrix
//!
//! Splits a CSR matrix into row-partitions so that each partition's data
//! is allocated (and thus NUMA-placed via first-touch) by the thread that
//! will later compute on it.  On multi-socket machines this keeps SpMV
//! memory traffic local.

use sonium_solvers_core::traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, SolverError, SolverFailure,
    SolverStatus,
};
use sonium_solvers_sys::topology::{
    AffinityStatus, MachineTopology, PlacementPlan, pin_current_thread_to,
};

use num_traits::Float;
use rayon::prelude::*;
use std::any::TypeId;
use thiserror::Error;

/// Errors reported by the checked NUMA SpMV entry points.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NumaSpmvError {
    #[error("NUMA SpMV input length mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid NUMA partition {partition}: {reason}")]
    InvalidPartition { partition: usize, reason: String },
    #[error("NUMA SpMV {operation} contains non-finite values")]
    NonFinite { operation: &'static str },
    #[error("NUMA SpMV worker panicked")]
    WorkerPanicked,
}

impl SolverFailure for NumaSpmvError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        match self {
            Self::DimensionMismatch { .. } => {
                sonium_solvers_core::traits::SolverErrorKind::DimensionMismatch
            }
            _ => sonium_solvers_core::traits::SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        SolverStatus::NumericalError
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

fn numa_error_to_solver_error(operation: &'static str, error: NumaSpmvError) -> SolverError {
    match error {
        NumaSpmvError::DimensionMismatch { expected, got } => {
            SolverError::DimensionMismatch { expected, got }
        }
        NumaSpmvError::InvalidPartition { partition, reason } => SolverError::OperationFailed {
            operation,
            reason: format!("partition {partition}: {reason}"),
        },
        NumaSpmvError::NonFinite { operation: source } => SolverError::OperationFailed {
            operation,
            reason: format!("{source} contains NaN or infinity"),
        },
        NumaSpmvError::WorkerPanicked => SolverError::OperationFailed {
            operation,
            reason: "NUMA worker panicked".into(),
        },
    }
}

/// One row-range partition of a CSR matrix.
///
/// All arrays are owned so the OS first-touch policy can place them on the
/// NUMA node of the thread that allocated them.
#[derive(Debug, Clone)]
pub struct CsrPartition<T: ComplexField> {
    /// First global row stored in this partition.
    pub row_start: usize,
    /// Number of rows in this partition.
    pub num_rows: usize,
    /// Local row pointers (length = `num_rows + 1`).
    /// `row_ptrs[0]` is always 0; entries are offsets into `col_indices`/`values`.
    pub row_ptrs: Vec<usize>,
    /// Column indices for every non-zero in this partition.
    pub col_indices: Vec<usize>,
    /// Values for every non-zero in this partition.
    pub values: Vec<T>,
    /// NUMA node that owns this partition, when topology detection was available.
    pub numa_node_id: Option<usize>,
    /// Socket that owns this partition, when topology detection was available.
    pub socket_id: Option<usize>,
    /// CPU set used for first-touch allocation and pinned applies.
    pub logical_cpus: Vec<usize>,
    /// Result of applying affinity before first-touch allocation.
    pub first_touch_affinity: AffinityStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionStats {
    pub row_start: usize,
    pub num_rows: usize,
    pub nnz: usize,
    pub numa_node_id: Option<usize>,
    pub socket_id: Option<usize>,
    pub logical_cpu_count: usize,
    pub first_touch_affinity_applied: bool,
    /// Exact result of the affinity request made before this partition was
    /// allocated.  The boolean above is retained as a compact compatibility
    /// field; this status is what evidence producers should serialize.
    pub first_touch_affinity: AffinityStatus,
}

/// Whether NUMA placement can be treated as evidence on the current host.
///
/// A UMA host may still use multiple workers and pinned policies for
/// functional testing, but elapsed time cannot prove locality there.  Such a
/// host is therefore explicitly `Unavailable`, rather than being inferred
/// from the presence of a non-empty CPU set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPlacementStatus {
    Unavailable,
    AffinityUnavailable,
    AffinityVerified,
}

impl NumaPlacementStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::AffinityUnavailable => "affinity_unavailable",
            Self::AffinityVerified => "affinity_verified",
        }
    }
}

/// Deterministic admission and work-balance evidence for a NUMA partitioning.
///
/// The report is deliberately independent of the host's measured bandwidth:
/// it records the requested versus admitted partition count, placement
/// availability, and the row/nonzero imbalance implied by the actual CSR
/// slices. Callers can therefore make an explicit fallback or oversubscription
/// decision before benchmarking or dispatching a solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumaPartitioningReport {
    pub requested_partitions: usize,
    pub admitted_partitions: usize,
    pub placement_workers: usize,
    pub num_rows: usize,
    pub total_nnz: usize,
    pub row_imbalance: f64,
    pub nnz_imbalance: f64,
    pub clamped_to_rows: bool,
    pub placement_available: bool,
    pub placement_status: NumaPlacementStatus,
}

/// Overflow-checked storage accounting for one NUMA SpMV policy.
///
/// `total_bytes` includes the partitioned CSR storage, the caller-owned input
/// vector, and any per-partition input replicas allocated by the replicated
/// policy. This is reporting/admission evidence; it does not retain the
/// temporary replicas after an apply returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumaStorageEstimate {
    pub partitioned_csr_bytes: usize,
    pub input_vector_bytes: usize,
    pub replicated_input_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputVectorPlacement {
    Shared,
    ReplicatedPerPartition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumaSpmvPolicy {
    pub pin_workers: bool,
    pub input_vector: InputVectorPlacement,
}

impl NumaSpmvPolicy {
    pub const SHARED_PINNED: Self = Self {
        pin_workers: true,
        input_vector: InputVectorPlacement::Shared,
    };

    pub const REPLICATED_PINNED: Self = Self {
        pin_workers: true,
        input_vector: InputVectorPlacement::ReplicatedPerPartition,
    };

    pub const SHARED_UNPINNED: Self = Self {
        pin_workers: false,
        input_vector: InputVectorPlacement::Shared,
    };
}

/// A CSR matrix split into NUMA-friendly row partitions.
///
/// Each partition owns its data independently, so first-touch allocation
/// places it on the NUMA node of the allocating thread.
///
/// Use [`NumaPartitionedCsr::from_csr`] to build, then call
/// [`NumaPartitionedCsr::matvec`] for parallel SpMV.
#[derive(Debug, Clone)]
pub struct NumaPartitionedCsr<T: ComplexField> {
    pub partitions: Vec<CsrPartition<T>>,
    pub num_rows: usize,
    pub num_cols: usize,
    requested_partitions: usize,
    placement_workers: usize,
}

/// Split contiguous CSR rows into partitions with approximately equal
/// nonzero work.
///
/// Row-count partitioning is a poor workload proxy for sparse matrices: one
/// long row can cost more than hundreds of short rows. The returned ranges
/// remain contiguous so each partition still owns one CSR slice, while the
/// cut points are chosen near equal prefix-work targets. A row containing a
/// disproportionate number of nonzeros is kept intact.
fn balanced_row_ranges_by_nnz(row_ptrs: &[usize], num_partitions: usize) -> Vec<(usize, usize)> {
    let num_rows = row_ptrs.len().saturating_sub(1);
    let num_partitions = num_partitions.max(1).min(num_rows.max(1));
    if num_rows == 0 {
        return Vec::new();
    }

    let total_nnz = row_ptrs[num_rows];
    if total_nnz == 0 {
        let base = num_rows / num_partitions;
        let remainder = num_rows % num_partitions;
        let mut ranges = Vec::with_capacity(num_partitions);
        let mut start = 0;
        for partition in 0..num_partitions {
            let rows = base + usize::from(partition < remainder);
            ranges.push((start, rows));
            start += rows;
        }
        return ranges;
    }

    let mut ranges = Vec::with_capacity(num_partitions);
    let mut start = 0;
    for partition in 0..num_partitions {
        let remaining_partitions = num_partitions - partition;
        let min_end = start + 1;
        let max_end = num_rows - (remaining_partitions - 1);
        let target = (total_nnz as u128 * (partition + 1) as u128) / num_partitions as u128;

        // Find the first legal row boundary at or beyond the target prefix.
        let mut low = min_end;
        let mut high = max_end;
        while low < high {
            let mid = low + (high - low) / 2;
            if (row_ptrs[mid] as u128) < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let upper = low;
        let lower = upper.saturating_sub(1).max(min_end);
        let lower_distance = (row_ptrs[lower] as u128).abs_diff(target);
        let upper_distance = (row_ptrs[upper] as u128).abs_diff(target);
        let end = if lower_distance <= upper_distance {
            lower
        } else {
            upper
        };

        ranges.push((start, end - start));
        start = end;
    }
    ranges
}

impl<T: ComplexField> NumaPartitionedCsr<T> {
    /// Build a NUMA-partitioned matrix from a regular [`CsrMatrix`].
    ///
    /// Each partition is allocated inside a rayon task so the OS first-touch
    /// policy places the memory near the core that will use it.
    pub fn from_csr(matrix: &super::CsrMatrix<T>, num_partitions: usize) -> Self {
        Self::try_from_csr(matrix, num_partitions)
            .unwrap_or_else(|error| panic!("invalid CSR matrix for NUMA partitioning: {error}"))
    }

    /// Build a NUMA-partitioned matrix after validating the mutable CSR
    /// storage and partition-count contract.
    pub fn try_from_csr(
        matrix: &super::CsrMatrix<T>,
        num_partitions: usize,
    ) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        if num_partitions == 0 {
            return Err(SolverError::OperationFailed {
                operation: "NUMA CSR construction",
                reason: "partition count must be greater than zero".into(),
            });
        }
        let topology = MachineTopology::detect();
        let plan = topology.placement_plan(num_partitions);
        Ok(Self::from_csr_with_placement_unchecked(matrix, &plan))
    }

    /// Estimate bytes copied into NUMA partition storage.
    ///
    /// The estimate covers partition row pointers, column indices, and value
    /// buffers (including one row-pointer boundary per admitted partition),
    /// but not the caller-owned source CSR. It is deterministic and can be
    /// used for admission before first-touch allocation.
    pub fn estimate_storage_bytes(
        matrix: &super::CsrMatrix<T>,
        num_partitions: usize,
    ) -> Result<usize, SolverError> {
        Ok(Self::estimate_storage_bytes_with_policy(
            matrix,
            num_partitions,
            NumaSpmvPolicy::SHARED_UNPINNED,
        )?
        .partitioned_csr_bytes)
    }

    /// Estimate all storage required by a NUMA policy before first-touch
    /// allocation.  Unlike [`Self::estimate_storage_bytes`], this includes
    /// the caller input vector and the per-partition replicas used by the
    /// replicated policy.
    pub fn estimate_storage_bytes_with_policy(
        matrix: &super::CsrMatrix<T>,
        num_partitions: usize,
        policy: NumaSpmvPolicy,
    ) -> Result<NumaStorageEstimate, SolverError> {
        matrix.validate_structure()?;
        if num_partitions == 0 {
            return Err(SolverError::OperationFailed {
                operation: "NUMA CSR policy memory estimate",
                reason: "partition count must be greater than zero".into(),
            });
        }
        let admitted = num_partitions.min(matrix.num_rows.max(1));
        let row_pointer_count =
            matrix
                .num_rows
                .checked_add(admitted)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation: "NUMA CSR policy memory estimate",
                    reason: "row-pointer count overflow".into(),
                })?;
        let index_bytes = matrix
            .col_indices
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| {
                row_pointer_count
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|rows| bytes.checked_add(rows))
            })
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "NUMA CSR policy memory estimate",
                reason: "index-byte count overflow".into(),
            })?;
        let value_bytes = matrix
            .values
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "NUMA CSR policy memory estimate",
                reason: "value-byte count overflow".into(),
            })?;
        let partitioned_csr_bytes =
            index_bytes
                .checked_add(value_bytes)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation: "NUMA CSR policy memory estimate",
                    reason: "total-byte count overflow".into(),
                })?;
        let input_vector_bytes = matrix
            .num_cols
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "NUMA CSR policy memory estimate",
                reason: "input-vector byte count overflow".into(),
            })?;
        let replicated_input_bytes = match policy.input_vector {
            InputVectorPlacement::Shared => 0,
            InputVectorPlacement::ReplicatedPerPartition => input_vector_bytes
                .checked_mul(admitted)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation: "NUMA CSR policy memory estimate",
                    reason: "replicated input-vector byte count overflow".into(),
                })?,
        };
        let total_bytes = partitioned_csr_bytes
            .checked_add(input_vector_bytes)
            .and_then(|bytes| bytes.checked_add(replicated_input_bytes))
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "NUMA CSR policy memory estimate",
                reason: "total-byte count overflow".into(),
            })?;
        Ok(NumaStorageEstimate {
            partitioned_csr_bytes,
            input_vector_bytes,
            replicated_input_bytes,
            total_bytes,
        })
    }

    /// Build NUMA partitions only when the requested storage fits a byte
    /// budget. Admission happens before any partition first-touch allocation.
    pub fn try_from_csr_with_memory_budget(
        matrix: &super::CsrMatrix<T>,
        num_partitions: usize,
        memory_budget_bytes: usize,
    ) -> Result<Self, SolverError> {
        let required = Self::estimate_storage_bytes(matrix, num_partitions)?;
        if required > memory_budget_bytes {
            return Err(SolverError::OperationFailed {
                operation: "NUMA CSR memory admission",
                reason: format!(
                    "required {required} bytes exceeds budget {memory_budget_bytes} bytes"
                ),
            });
        }
        Self::try_from_csr(matrix, num_partitions)
    }

    /// Build with an explicit placement plan only when the copied partition
    /// storage fits the supplied byte budget. The checked estimate runs before
    /// any worker is spawned or any partition is first-touched.
    pub fn try_from_csr_with_placement_and_memory_budget(
        matrix: &super::CsrMatrix<T>,
        placement: &PlacementPlan,
        memory_budget_bytes: usize,
    ) -> Result<Self, SolverError> {
        let required = Self::estimate_storage_bytes(matrix, placement.workers.len())?;
        if required > memory_budget_bytes {
            return Err(SolverError::OperationFailed {
                operation: "NUMA CSR placement memory admission",
                reason: format!(
                    "required {required} bytes exceeds budget {memory_budget_bytes} bytes"
                ),
            });
        }
        Self::try_from_csr_with_placement(matrix, placement)
    }

    /// Build with a placement plan only when the complete selected policy
    /// fits the byte budget.  In particular, replicated input storage is
    /// admitted before a worker is spawned or any partition is first-touched.
    pub fn try_from_csr_with_placement_and_policy_memory_budget(
        matrix: &super::CsrMatrix<T>,
        placement: &PlacementPlan,
        policy: NumaSpmvPolicy,
        memory_budget_bytes: usize,
    ) -> Result<Self, SolverError> {
        let estimate =
            Self::estimate_storage_bytes_with_policy(matrix, placement.workers.len(), policy)?;
        if estimate.total_bytes > memory_budget_bytes {
            return Err(SolverError::OperationFailed {
                operation: "NUMA CSR policy placement memory admission",
                reason: format!(
                    "required {} bytes exceeds budget {memory_budget_bytes} bytes",
                    estimate.total_bytes
                ),
            });
        }
        Self::try_from_csr_with_placement(matrix, placement)
    }

    /// Build with an explicit placement plan. Linux workers are pinned before
    /// allocation so matrix partition storage is first-touched on the owner CPU set.
    pub fn from_csr_with_placement(
        matrix: &super::CsrMatrix<T>,
        placement: &PlacementPlan,
    ) -> Self {
        Self::try_from_csr_with_placement(matrix, placement)
            .unwrap_or_else(|error| panic!("invalid CSR matrix for NUMA placement: {error}"))
    }

    /// Build with an explicit placement plan after validating mutable CSR
    /// storage. The placement plan itself is still supplied by the caller.
    pub fn try_from_csr_with_placement(
        matrix: &super::CsrMatrix<T>,
        placement: &PlacementPlan,
    ) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        Ok(Self::from_csr_with_placement_unchecked(matrix, placement))
    }

    fn from_csr_with_placement_unchecked(
        matrix: &super::CsrMatrix<T>,
        placement: &PlacementPlan,
    ) -> Self {
        let n = matrix.num_rows;
        let requested = placement.workers.len().max(1);
        let num_partitions = requested.max(1).min(n.max(1));

        let ranges = balanced_row_ranges_by_nnz(&matrix.row_ptrs, num_partitions);

        let mut partitions = std::thread::scope(|scope| {
            let handles = ranges
                .into_iter()
                .enumerate()
                .map(|(partition_index, (row_start, num_rows))| {
                    let worker = placement.worker_for_index(partition_index).cloned();
                    scope.spawn(move || {
                        let affinity = worker
                            .as_ref()
                            .map(|worker| pin_current_thread_to(&worker.logical_cpus))
                            .unwrap_or(AffinityStatus::Unsupported {
                                reason: "no worker placement for partition".to_string(),
                            });

                        let global_start_ptr = matrix.row_ptrs[row_start];
                        let global_end_ptr = matrix.row_ptrs[row_start + num_rows];
                        let nnz = global_end_ptr - global_start_ptr;

                        let mut row_ptrs = Vec::with_capacity(num_rows + 1);
                        for r in 0..=num_rows {
                            row_ptrs.push(matrix.row_ptrs[row_start + r] - global_start_ptr);
                        }

                        let col_indices =
                            matrix.col_indices[global_start_ptr..global_start_ptr + nnz].to_vec();
                        let values =
                            matrix.values[global_start_ptr..global_start_ptr + nnz].to_vec();

                        CsrPartition {
                            row_start,
                            num_rows,
                            row_ptrs,
                            col_indices,
                            values,
                            numa_node_id: worker.as_ref().map(|worker| worker.numa_node_id),
                            socket_id: worker.as_ref().map(|worker| worker.socket_id),
                            logical_cpus: worker
                                .as_ref()
                                .map(|worker| worker.logical_cpus.clone())
                                .unwrap_or_default(),
                            first_touch_affinity: affinity,
                        }
                    })
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .map(|handle| handle.join().expect("NUMA partition worker panicked"))
                .collect::<Vec<_>>()
        });
        partitions.sort_by_key(|partition| partition.row_start);

        Self {
            partitions,
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            requested_partitions: placement.workers.len(),
            placement_workers: placement.workers.len(),
        }
    }

    /// Number of non-zeros across all partitions.
    pub fn nnz(&self) -> usize {
        self.partitions.iter().map(|p| p.values.len()).sum()
    }

    /// Number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Lightweight partition metadata for benchmark/report diagnostics.
    pub fn partition_stats(&self) -> Vec<PartitionStats> {
        self.partitions
            .iter()
            .map(|partition| PartitionStats {
                row_start: partition.row_start,
                num_rows: partition.num_rows,
                nnz: partition.values.len(),
                numa_node_id: partition.numa_node_id,
                socket_id: partition.socket_id,
                logical_cpu_count: partition.logical_cpus.len(),
                first_touch_affinity_applied: partition.first_touch_affinity.is_applied(),
                first_touch_affinity: partition.first_touch_affinity.clone(),
            })
            .collect()
    }

    /// Report deterministic partition admission and work balance.
    pub fn partitioning_report(&self) -> NumaPartitioningReport {
        let stats = self.partition_stats();
        let admitted_partitions = stats.len();
        let total_nnz = stats.iter().map(|stat| stat.nnz).sum::<usize>();
        let max_rows = stats.iter().map(|stat| stat.num_rows).max().unwrap_or(0);
        let max_nnz = stats.iter().map(|stat| stat.nnz).max().unwrap_or(0);
        let row_mean = if admitted_partitions == 0 {
            0.0
        } else {
            self.num_rows as f64 / admitted_partitions as f64
        };
        let nnz_mean = if admitted_partitions == 0 {
            0.0
        } else {
            total_nnz as f64 / admitted_partitions as f64
        };
        let row_imbalance = if row_mean > 0.0 {
            max_rows as f64 / row_mean - 1.0
        } else {
            0.0
        };
        let nnz_imbalance = if nnz_mean > 0.0 {
            max_nnz as f64 / nnz_mean - 1.0
        } else {
            0.0
        };
        let placement_status = placement_status(&stats);
        NumaPartitioningReport {
            requested_partitions: self.requested_partitions,
            admitted_partitions,
            placement_workers: self.placement_workers,
            num_rows: self.num_rows,
            total_nnz,
            row_imbalance,
            nnz_imbalance,
            clamped_to_rows: self.requested_partitions > self.num_rows.max(1),
            placement_available: placement_status != NumaPlacementStatus::Unavailable,
            placement_status,
        }
    }

    /// Estimate the actual copied storage used by this partitioned matrix and
    /// the selected input-vector placement policy.
    pub fn storage_estimate(
        &self,
        policy: NumaSpmvPolicy,
    ) -> Result<NumaStorageEstimate, SolverError> {
        let value_bytes = std::mem::size_of::<T>();
        let index_bytes = std::mem::size_of::<usize>();
        let partitioned_csr_bytes = self
            .partitions
            .iter()
            .try_fold(0usize, |total, part| {
                let values = part.values.len().checked_mul(value_bytes);
                let columns = part.col_indices.len().checked_mul(index_bytes);
                let row_ptrs = part.row_ptrs.len().checked_mul(index_bytes);
                values
                    .and_then(|bytes| columns.and_then(|columns| bytes.checked_add(columns)))
                    .and_then(|bytes| row_ptrs.and_then(|row_ptrs| bytes.checked_add(row_ptrs)))
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "NUMA CSR storage estimate",
                reason: "partitioned CSR byte count overflow".into(),
            })?;
        let input_vector_bytes =
            self.num_cols
                .checked_mul(value_bytes)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation: "NUMA CSR storage estimate",
                    reason: "input-vector byte count overflow".into(),
                })?;
        let replicated_input_bytes = match policy.input_vector {
            InputVectorPlacement::Shared => 0,
            InputVectorPlacement::ReplicatedPerPartition => input_vector_bytes
                .checked_mul(self.partitions.len())
                .ok_or_else(|| SolverError::OperationFailed {
                    operation: "NUMA CSR storage estimate",
                    reason: "replicated input-vector byte count overflow".into(),
                })?,
        };
        let total_bytes = partitioned_csr_bytes
            .checked_add(input_vector_bytes)
            .and_then(|bytes| bytes.checked_add(replicated_input_bytes))
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "NUMA CSR storage estimate",
                reason: "total NUMA storage byte count overflow".into(),
            })?;
        Ok(NumaStorageEstimate {
            partitioned_csr_bytes,
            input_vector_bytes,
            replicated_input_bytes,
            total_bytes,
        })
    }

    /// Validate the partition invariants used by the checked SpMV paths.
    fn validate_partitions(&self) -> Result<(), NumaSpmvError> {
        self.validate_hot_partitions()?;
        for partition in self.partitions.iter() {
            if partition
                .values
                .iter()
                .any(|value| !value.norm().is_finite())
            {
                return Err(NumaSpmvError::NonFinite {
                    operation: "matrix values",
                });
            }
        }
        Ok(())
    }

    /// Lightweight structural check for the infallible NUMA hot path.
    ///
    /// Full [`validate_partitions`](Self::validate_partitions) additionally
    /// scans every partition value for finiteness — a second full streaming
    /// pass over the matrix data with no memory-safety effect (non-finite
    /// values propagate arithmetically; they cannot cause out-of-bounds
    /// access). The hot path only needs the memory-safety invariants:
    /// covering row ranges, per-partition CSR shape, monotone row pointers,
    /// length agreement, and in-bounds columns. This mirrors CSR's
    /// `validate_hot_structure` split: infallible entry points propagate
    /// non-finite values arithmetically, while the `try_*` boundaries still
    /// reject them.
    fn validate_hot_partitions(&self) -> Result<(), NumaSpmvError> {
        let mut next_row = 0;
        for (partition_index, partition) in self.partitions.iter().enumerate() {
            if partition.row_start != next_row {
                return Err(NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: format!(
                        "row range starts at {}, expected {next_row}",
                        partition.row_start
                    ),
                });
            }
            let expected_row_ptrs = partition.num_rows.checked_add(1).ok_or_else(|| {
                NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: "row count overflows row-pointer length".to_string(),
                }
            })?;
            if partition.row_ptrs.len() != expected_row_ptrs {
                return Err(NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: format!(
                        "row pointer length {} does not equal rows + 1 ({})",
                        partition.row_ptrs.len(),
                        expected_row_ptrs
                    ),
                });
            }
            if partition.row_ptrs.first().copied().unwrap_or(usize::MAX) != 0 {
                return Err(NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: "first row pointer is not zero".to_string(),
                });
            }
            if partition
                .row_ptrs
                .windows(2)
                .any(|window| window[0] > window[1])
            {
                return Err(NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: "row pointers are not monotonic".to_string(),
                });
            }
            if partition.row_ptrs.last().copied().unwrap_or_default() != partition.values.len()
                || partition.values.len() != partition.col_indices.len()
            {
                return Err(NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: "CSR value, column, and row-pointer lengths disagree".to_string(),
                });
            }
            if partition
                .col_indices
                .iter()
                .any(|&column| column >= self.num_cols)
            {
                return Err(NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: "column index exceeds matrix width".to_string(),
                });
            }
            next_row = next_row.checked_add(partition.num_rows).ok_or_else(|| {
                NumaSpmvError::InvalidPartition {
                    partition: partition_index,
                    reason: "row range overflows usize".to_string(),
                }
            })?;
        }
        if next_row != self.num_rows {
            return Err(NumaSpmvError::InvalidPartition {
                partition: self.partitions.len(),
                reason: format!(
                    "partitions cover {next_row} rows, expected {}",
                    self.num_rows
                ),
            });
        }
        Ok(())
    }

    /// Checked NUMA-aware SpMV with explicit worker/input placement policy.
    ///
    /// This entry point reports invalid partition structure and
    /// converts scoped-worker failures into a typed
    /// error instead of panicking after a partial computation.
    pub fn try_matvec_with_policy(
        &self,
        x: &[T],
        policy: NumaSpmvPolicy,
    ) -> Result<Vec<T>, NumaSpmvError>
    where
        T: 'static,
    {
        if x.len() != self.num_cols {
            return Err(NumaSpmvError::DimensionMismatch {
                expected: self.num_cols,
                got: x.len(),
            });
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            return Err(NumaSpmvError::NonFinite { operation: "input" });
        }
        self.validate_partitions()?;

        let chunks = std::thread::scope(|scope| {
            let handles = self
                .partitions
                .iter()
                .map(|part| {
                    scope.spawn(move || {
                        if policy.pin_workers {
                            let _ = pin_current_thread_to(&part.logical_cpus);
                        }
                        match policy.input_vector {
                            InputVectorPlacement::Shared => (
                                part.row_start,
                                matvec_partition(part, x, self.num_cols <= i32::MAX as usize),
                            ),
                            InputVectorPlacement::ReplicatedPerPartition => {
                                let local_x = x.to_vec();
                                (
                                    part.row_start,
                                    matvec_partition(
                                        part,
                                        &local_x,
                                        self.num_cols <= i32::MAX as usize,
                                    ),
                                )
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| NumaSpmvError::WorkerPanicked))
                .collect::<Result<Vec<_>, _>>()
        })?;

        let mut y = vec![T::zero(); self.num_rows];
        for (row_start, local_y) in chunks {
            for (i, val) in local_y.into_iter().enumerate() {
                y[row_start + i] = val;
            }
        }
        if y.iter().any(|value| !value.norm().is_finite()) {
            return Err(NumaSpmvError::NonFinite {
                operation: "output",
            });
        }
        Ok(y)
    }

    /// Checked shared-input NUMA SpMV using unpinned workers.
    pub fn try_matvec(&self, x: &[T]) -> Result<Vec<T>, NumaSpmvError>
    where
        T: 'static,
    {
        self.try_matvec_with_policy(x, NumaSpmvPolicy::SHARED_UNPINNED)
    }

    /// Checked transpose SpMV. Public partition metadata is validated before
    /// the parallel transpose kernel reads row pointers or column indices.
    pub fn try_matvec_transpose(&self, x: &[T]) -> Result<Vec<T>, NumaSpmvError> {
        self.try_matvec_transpose_with_policy(x, NumaSpmvPolicy::SHARED_UNPINNED)
    }

    /// Checked transpose SpMV with explicit worker/input placement policy.
    pub fn try_matvec_transpose_with_policy(
        &self,
        x: &[T],
        policy: NumaSpmvPolicy,
    ) -> Result<Vec<T>, NumaSpmvError> {
        if x.len() != self.num_rows {
            return Err(NumaSpmvError::DimensionMismatch {
                expected: self.num_rows,
                got: x.len(),
            });
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            return Err(NumaSpmvError::NonFinite {
                operation: "transpose input",
            });
        }
        self.validate_partitions()?;
        let y = self.matvec_adjoint_with_policy(x, policy, false);
        if y.iter().any(|value| !value.norm().is_finite()) {
            return Err(NumaSpmvError::NonFinite {
                operation: "transpose output",
            });
        }
        Ok(y)
    }

    /// Checked Hermitian transpose SpMV. Public partition metadata is
    /// validated before the parallel Hermitian kernel reads the layout.
    pub fn try_matvec_hermitian(&self, x: &[T]) -> Result<Vec<T>, NumaSpmvError> {
        self.try_matvec_hermitian_with_policy(x, NumaSpmvPolicy::SHARED_UNPINNED)
    }

    /// Checked Hermitian transpose SpMV with explicit worker/input placement
    /// policy.
    pub fn try_matvec_hermitian_with_policy(
        &self,
        x: &[T],
        policy: NumaSpmvPolicy,
    ) -> Result<Vec<T>, NumaSpmvError> {
        if x.len() != self.num_rows {
            return Err(NumaSpmvError::DimensionMismatch {
                expected: self.num_rows,
                got: x.len(),
            });
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            return Err(NumaSpmvError::NonFinite {
                operation: "Hermitian input",
            });
        }
        self.validate_partitions()?;
        let y = self.matvec_adjoint_with_policy(x, policy, true);
        if y.iter().any(|value| !value.norm().is_finite()) {
            return Err(NumaSpmvError::NonFinite {
                operation: "Hermitian output",
            });
        }
        Ok(y)
    }

    /// Parallel matrix-vector product with explicit NUMA policy.
    ///
    /// When `pin_workers` is enabled, each partition apply runs on a scoped
    /// worker pinned to that partition's CPU set. With replicated input, each
    /// worker clones `x` after pinning so the replica is first-touched on the
    /// owner CPUs; this costs O(n * partitions) memory and is only worthwhile
    /// for very large matrices with broad remote reads.
    pub fn matvec_with_policy(&self, x: &[T], policy: NumaSpmvPolicy) -> Vec<T> {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");

        if !policy.pin_workers && policy.input_vector == InputVectorPlacement::Shared {
            return self.matvec(x);
        }

        self.validate_hot_partitions()
            .expect("invalid NUMA CSR storage");

        let chunks = std::thread::scope(|scope| {
            let handles = self
                .partitions
                .iter()
                .map(|part| {
                    scope.spawn(move || {
                        if policy.pin_workers {
                            let _ = pin_current_thread_to(&part.logical_cpus);
                        }
                        match policy.input_vector {
                            InputVectorPlacement::Shared => (
                                part.row_start,
                                matvec_partition(part, x, self.num_cols <= i32::MAX as usize),
                            ),
                            InputVectorPlacement::ReplicatedPerPartition => {
                                let local_x = x.to_vec();
                                (
                                    part.row_start,
                                    matvec_partition(
                                        part,
                                        &local_x,
                                        self.num_cols <= i32::MAX as usize,
                                    ),
                                )
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .map(|handle| handle.join().expect("NUMA SpMV worker panicked"))
                .collect::<Vec<_>>()
        });

        let mut y = vec![T::zero(); self.num_rows];
        for (row_start, local_y) in chunks {
            for (i, val) in local_y.into_iter().enumerate() {
                y[row_start + i] = val;
            }
        }
        y
    }

    /// Parallel matrix-vector product: y = A * x
    ///
    /// Each partition computes its row range independently — no locking,
    /// no overlap between writers. For `f64` matrices, each partition uses
    /// SIMD (AVX2/NEON) internally.
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        // Public partition metadata can be mutated after construction. In
        // particular, prove disjoint output ranges before making raw slices.
        // Hot-path validation only (memory-safety invariants); non-finite
        // values propagate arithmetically, mirroring the CSR hot path.
        self.validate_hot_partitions()
            .expect("invalid NUMA CSR storage");

        let x_slice = x;

        sonium_solvers_sys::runtime::with_bandwidth_pool(|| {
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                let mut y = vec![T::zero(); self.num_rows];
                let y_slice = y.as_mut_slice();

                // SAFETY: T == f64 verified above. Pointer casts are layout-identical.
                let x_f64 = unsafe { &*(x_slice as *const [T] as *const [f64]) };

                // Store as usize so we can share across rayon tasks.
                // SAFETY: partitions write to disjoint row ranges of the y buffer
                // so there are no data races.
                let y_base = y_slice.as_mut_ptr() as usize;

                self.partitions.par_iter().for_each(|part| {
                    let values_f64 =
                        unsafe { &*(part.values.as_slice() as *const [T] as *const [f64]) };
                    let local_y = unsafe {
                        std::slice::from_raw_parts_mut(
                            (y_base as *mut f64).add(part.row_start),
                            part.num_rows,
                        )
                    };
                    unsafe {
                        super::csr::spmv_f64_simd(
                            values_f64,
                            &part.col_indices,
                            &part.row_ptrs,
                            x_f64,
                            local_y,
                            part.num_rows,
                            self.num_cols <= i32::MAX as usize,
                        );
                    }
                });

                return y;
            }

            // Generic scalar fallback for all other types
            let chunks: Vec<(usize, Vec<T>)> = self
                .partitions
                .par_iter()
                .map(|part| {
                    let local_y: Vec<T> = (0..part.num_rows)
                        .map(|i| {
                            let mut sum = T::zero();
                            for idx in part.row_ptrs[i]..part.row_ptrs[i + 1] {
                                sum += part.values[idx] * x_slice[part.col_indices[idx]];
                            }
                            sum
                        })
                        .collect();
                    (part.row_start, local_y)
                })
                .collect();

            let mut y = vec![T::zero(); self.num_rows];
            for (row_start, local_y) in chunks {
                for (i, val) in local_y.into_iter().enumerate() {
                    y[row_start + i] = val;
                }
            }
            y
        })
    }

    /// Transpose matrix-vector product: y = A^T * x
    pub fn matvec_transpose(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");

        self.matvec_transpose_with_policy(x, NumaSpmvPolicy::SHARED_UNPINNED)
    }

    /// Transpose matrix-vector product with explicit worker/input placement.
    pub fn matvec_transpose_with_policy(&self, x: &[T], policy: NumaSpmvPolicy) -> Vec<T> {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        self.matvec_adjoint_with_policy(x, policy, false)
    }

    fn matvec_adjoint_with_policy(
        &self,
        x: &[T],
        policy: NumaSpmvPolicy,
        conjugate: bool,
    ) -> Vec<T> {
        // Hot-path validation only (memory-safety invariants); non-finite
        // values propagate arithmetically, mirroring the CSR hot path.
        self.validate_hot_partitions()
            .expect("invalid NUMA CSR storage");
        if !policy.pin_workers && policy.input_vector == InputVectorPlacement::Shared {
            return sonium_solvers_sys::runtime::with_bandwidth_pool(|| {
                let partials: Vec<Vec<T>> = self
                    .partitions
                    .par_iter()
                    .map(|part| {
                        let mut local_y = vec![T::zero(); self.num_cols];
                        for i in 0..part.num_rows {
                            let global_row = part.row_start + i;
                            let xi = x[global_row];
                            let start = part.row_ptrs[i];
                            let end = part.row_ptrs[i + 1];
                            for idx in start..end {
                                let j = part.col_indices[idx];
                                let value = if conjugate {
                                    part.values[idx].conj()
                                } else {
                                    part.values[idx]
                                };
                                local_y[j] += value * xi;
                            }
                        }
                        local_y
                    })
                    .collect();

                let mut y = vec![T::zero(); self.num_cols];
                for partial in partials {
                    for (j, val) in partial.into_iter().enumerate() {
                        y[j] += val;
                    }
                }
                y
            });
        }

        // Each partition contributes to potentially all columns, so we need
        // per-partition partial results then reduce.
        let partials = std::thread::scope(|scope| {
            let handles = self
                .partitions
                .iter()
                .map(|part| {
                    scope.spawn(move || {
                        if policy.pin_workers {
                            let _ = pin_current_thread_to(&part.logical_cpus);
                        }
                        let local_x = match policy.input_vector {
                            InputVectorPlacement::Shared => None,
                            InputVectorPlacement::ReplicatedPerPartition => Some(x.to_vec()),
                        };
                        let input = local_x.as_deref().unwrap_or(x);
                        let mut local_y = vec![T::zero(); self.num_cols];
                        for i in 0..part.num_rows {
                            let global_row = part.row_start + i;
                            let xi = input[global_row];
                            let start = part.row_ptrs[i];
                            let end = part.row_ptrs[i + 1];
                            for idx in start..end {
                                let j = part.col_indices[idx];
                                let value = if conjugate {
                                    part.values[idx].conj()
                                } else {
                                    part.values[idx]
                                };
                                local_y[j] += value * xi;
                            }
                        }
                        local_y
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("NUMA transpose worker panicked"))
                .collect::<Vec<_>>()
        });

        let mut y = vec![T::zero(); self.num_cols];
        for partial in partials {
            for (j, val) in partial.into_iter().enumerate() {
                y[j] += val;
            }
        }
        y
    }

    /// Hermitian (conjugate transpose) matrix-vector product: y = A^H * x
    pub fn matvec_hermitian(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        self.matvec_hermitian_with_policy(x, NumaSpmvPolicy::SHARED_UNPINNED)
    }

    /// Hermitian matrix-vector product with explicit worker/input placement.
    pub fn matvec_hermitian_with_policy(&self, x: &[T], policy: NumaSpmvPolicy) -> Vec<T> {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        self.matvec_adjoint_with_policy(x, policy, true)
    }
}

fn placement_status(stats: &[PartitionStats]) -> NumaPlacementStatus {
    if stats.is_empty()
        || stats.iter().any(|stat| {
            stat.numa_node_id.is_none() || stat.socket_id.is_none() || stat.logical_cpu_count == 0
        })
    {
        return NumaPlacementStatus::Unavailable;
    }

    let node_count = stats
        .iter()
        .filter_map(|stat| stat.numa_node_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let socket_count = stats
        .iter()
        .filter_map(|stat| stat.socket_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if node_count < 2 && socket_count < 2 {
        return NumaPlacementStatus::Unavailable;
    }
    if stats.iter().all(|stat| stat.first_touch_affinity_applied) {
        NumaPlacementStatus::AffinityVerified
    } else {
        NumaPlacementStatus::AffinityUnavailable
    }
}

fn matvec_partition<T: ComplexField + 'static>(
    part: &CsrPartition<T>,
    x: &[T],
    allow_avx2_gather: bool,
) -> Vec<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let mut local_y = vec![T::zero(); part.num_rows];
        let values_f64 = unsafe { &*(part.values.as_slice() as *const [T] as *const [f64]) };
        let x_f64 = unsafe { &*(x as *const [T] as *const [f64]) };
        let local_y_f64 = unsafe { &mut *(local_y.as_mut_slice() as *mut [T] as *mut [f64]) };
        unsafe {
            super::csr::spmv_f64_simd(
                values_f64,
                &part.col_indices,
                &part.row_ptrs,
                x_f64,
                local_y_f64,
                part.num_rows,
                allow_avx2_gather,
            );
        }
        return local_y;
    }

    (0..part.num_rows)
        .map(|i| {
            let mut sum = T::zero();
            for idx in part.row_ptrs[i]..part.row_ptrs[i + 1] {
                sum += part.values[idx] * x[part.col_indices[idx]];
            }
            sum
        })
        .collect()
}

impl<T: ComplexField> LinearOperator<T> for NumaPartitionedCsr<T> {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        self.matvec(x)
    }

    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec(x)
            .map_err(|error| numa_error_to_solver_error("NUMA CSR SpMV", error))
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        let result = self.matvec(x);
        y.copy_from_slice(&result);
    }

    fn apply_transpose(&self, x: &[T]) -> Vec<T> {
        self.matvec_transpose(x)
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_transpose(x)
            .map_err(|error| numa_error_to_solver_error("NUMA CSR transpose SpMV", error))
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        let result = self.matvec_transpose(x);
        y.copy_from_slice(&result);
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        self.matvec_hermitian(x)
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_hermitian(x)
            .map_err(|error| numa_error_to_solver_error("NUMA CSR Hermitian SpMV", error))
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// Heuristic for the number of NUMA partitions on this machine.
///
/// On Linux reads `/sys/devices/system/node/` to count NUMA nodes.
/// On macOS (UMA) or unknown platforms, returns available parallelism / 16
/// (clamped to at least 1).
pub fn detect_numa_partitions() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            let count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("node"))
                })
                .count();
            if count > 0 {
                return count;
            }
        }
    }

    // macOS is UMA — partition by available parallelism for cache locality
    let cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (cpus / 16).max(1)
}

// Convenience method on CsrMatrix
impl<T: ComplexField> super::CsrMatrix<T> {
    /// Convert to a NUMA-partitioned representation.
    ///
    /// `num_partitions` should match the number of NUMA nodes (or use
    /// [`detect_numa_partitions`] for an automatic heuristic).
    pub fn to_numa_partitioned(&self, num_partitions: usize) -> NumaPartitionedCsr<T> {
        NumaPartitionedCsr::from_csr(self, num_partitions)
    }

    /// Checked variant of [`Self::to_numa_partitioned`].
    pub fn try_to_numa_partitioned(
        &self,
        num_partitions: usize,
    ) -> Result<NumaPartitionedCsr<T>, SolverError> {
        NumaPartitionedCsr::try_from_csr(self, num_partitions)
    }

    pub fn to_numa_partitioned_with_placement(
        &self,
        placement: &PlacementPlan,
    ) -> NumaPartitionedCsr<T> {
        NumaPartitionedCsr::from_csr_with_placement(self, placement)
    }

    /// Checked variant of [`Self::to_numa_partitioned_with_placement`].
    pub fn try_to_numa_partitioned_with_placement(
        &self,
        placement: &PlacementPlan,
    ) -> Result<NumaPartitionedCsr<T>, SolverError> {
        NumaPartitionedCsr::try_from_csr_with_placement(self, placement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrMatrix;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use sonium_solvers_core::dense::DenseMatrix;

    fn dense2<T: sonium_solvers_core::traits::ComplexField>(
        rows: usize,
        cols: usize,
        data: Vec<T>,
    ) -> DenseMatrix<T> {
        DenseMatrix::from_row_major_vec(rows, cols, data)
    }

    #[test]
    fn numa_ranges_balance_nonzero_work_for_skewed_rows() {
        // Seven unit rows followed by one seven-entry row. Equal row counts
        // would produce loads 4 and 10; prefix-work cuts produce 7 and 7.
        let row_ptrs = [0, 1, 2, 3, 4, 5, 6, 7, 14];
        let ranges = balanced_row_ranges_by_nnz(&row_ptrs, 2);

        assert_eq!(ranges, vec![(0, 7), (7, 1)]);
        assert_eq!(
            ranges
                .iter()
                .map(|&(start, rows)| row_ptrs[start + rows] - row_ptrs[start])
                .collect::<Vec<_>>(),
            vec![7, 7]
        );
        assert_eq!(ranges[0].0 + ranges[0].1, ranges[1].0);
        assert_eq!(ranges[1].0 + ranges[1].1, row_ptrs.len() - 1);
    }

    #[test]
    fn checked_numa_constructor_rejects_public_csr_mutation() {
        let mut matrix =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]);
        matrix.row_ptrs[2] = 1;

        let error = NumaPartitionedCsr::try_from_csr(&matrix, 1).unwrap_err();
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "CSR validation",
                ..
            }
        ));
    }

    #[test]
    fn checked_numa_constructor_rejects_zero_partitions() {
        let matrix = CsrMatrix::<f64>::new(2, 2);
        let error = NumaPartitionedCsr::try_from_csr(&matrix, 0).unwrap_err();
        assert!(matches!(
            error,
            sonium_solvers_core::traits::SolverError::OperationFailed {
                operation: "NUMA CSR construction",
                ..
            }
        ));
    }

    #[test]
    fn test_numa_matvec_matches_csr() {
        let dense = dense2(
            3,
            3,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(5.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];

        let y_csr = csr.matvec(&x);
        let y_numa = numa.matvec(&x);

        for i in 0..3 {
            assert_relative_eq!((y_csr[i] - y_numa[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_numa_transpose_matches_csr() {
        let dense = dense2(
            3,
            2,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -0.5),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];

        let y_csr = csr.matvec_transpose(&x);
        let y_numa = numa.matvec_transpose(&x);

        for i in 0..2 {
            assert_relative_eq!((y_csr[i] - y_numa[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_numa_hermitian_matches_csr() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);

        let x = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        let y_csr = csr.matvec_hermitian(&x);
        let y_numa = numa.matvec_hermitian(&x);

        for i in 0..2 {
            assert_relative_eq!((y_csr[i] - y_numa[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_numa_single_partition() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(1);

        assert_eq!(numa.num_partitions(), 1);
        assert_eq!(numa.nnz(), csr.nnz());

        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let y = numa.matvec(&x);
        assert_relative_eq!(y[0].re, 3.0, epsilon = 1e-12);
        assert_relative_eq!(y[1].re, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_numa_many_partitions() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(10);

        assert!(numa.num_partitions() <= 2);

        let x = vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];
        let y = numa.matvec(&x);
        assert_relative_eq!(y[0].re, 3.0, epsilon = 1e-12);
        assert_relative_eq!(y[1].re, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_numa_f64() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);

        let x = vec![1.0_f64, 2.0];
        let y_csr = csr.matvec(&x);
        let y_numa = numa.matvec(&x);

        assert_relative_eq!(y_csr[0], y_numa[0], epsilon = 1e-12);
        assert_relative_eq!(y_csr[1], y_numa[1], epsilon = 1e-12);
    }

    #[test]
    fn test_numa_policy_matches_default_matvec() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 5.0],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);
        let x = vec![1.0_f64, 2.0, 3.0];

        let y_default = numa.matvec(&x);
        let y_shared = numa.matvec_with_policy(&x, NumaSpmvPolicy::SHARED_PINNED);
        let y_replicated = numa.matvec_with_policy(&x, NumaSpmvPolicy::REPLICATED_PINNED);

        for i in 0..3 {
            assert_relative_eq!(y_default[i], y_shared[i], epsilon = 1e-12);
            assert_relative_eq!(y_default[i], y_replicated[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn storage_estimate_accounts_for_replicated_inputs() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 5.0],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);
        let shared = numa
            .storage_estimate(NumaSpmvPolicy::SHARED_PINNED)
            .expect("shared storage estimate");
        let replicated = numa
            .storage_estimate(NumaSpmvPolicy::REPLICATED_PINNED)
            .expect("replicated storage estimate");

        assert_eq!(shared.replicated_input_bytes, 0);
        assert_eq!(
            replicated.replicated_input_bytes,
            replicated.input_vector_bytes * numa.num_partitions()
        );
        assert_eq!(
            replicated.total_bytes,
            shared.total_bytes + replicated.replicated_input_bytes
        );
    }

    #[test]
    fn policy_memory_estimate_accounts_for_replicas_before_admission() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 5.0],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let shared = NumaPartitionedCsr::estimate_storage_bytes_with_policy(
            &csr,
            2,
            NumaSpmvPolicy::SHARED_PINNED,
        )
        .expect("shared policy estimate");
        let replicated = NumaPartitionedCsr::estimate_storage_bytes_with_policy(
            &csr,
            2,
            NumaSpmvPolicy::REPLICATED_PINNED,
        )
        .expect("replicated policy estimate");

        assert_eq!(shared.replicated_input_bytes, 0);
        assert_eq!(
            replicated.replicated_input_bytes,
            replicated.input_vector_bytes * 2
        );
        assert_eq!(
            replicated.total_bytes,
            shared.total_bytes + replicated.replicated_input_bytes
        );
        let placement = MachineTopology::detect().placement_plan(2);
        let admitted = NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
            &csr,
            &placement,
            NumaSpmvPolicy::REPLICATED_PINNED,
            replicated.total_bytes,
        )
        .expect("exact policy budget should admit");
        assert_eq!(admitted.nnz(), csr.nnz());
        let error = NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
            &csr,
            &placement,
            NumaSpmvPolicy::REPLICATED_PINNED,
            replicated.total_bytes - 1,
        )
        .expect_err("one byte short must reject before first touch");
        assert!(matches!(
            error,
            SolverError::OperationFailed {
                operation: "NUMA CSR policy placement memory admission",
                ..
            }
        ));
    }

    #[test]
    fn checked_numa_matvec_matches_policy_paths() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 5.0],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);
        let x = vec![1.0_f64, 2.0, 3.0];

        let expected = numa.matvec(&x);
        let checked = numa
            .try_matvec(&x)
            .expect("valid NUMA partitions should use the checked path");
        let checked_replicated = numa
            .try_matvec_with_policy(&x, NumaSpmvPolicy::REPLICATED_PINNED)
            .expect("valid replicated NUMA input should use the checked path");

        assert_eq!(checked, expected);
        assert_eq!(checked_replicated, expected);
    }

    #[test]
    fn checked_numa_adjoint_policy_paths_match_csr() {
        let dense = dense2(
            3,
            2,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -0.5),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);
        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let expected_transpose = csr.matvec_transpose(&x);
        let expected_hermitian = csr.matvec_hermitian(&x);
        for policy in [
            NumaSpmvPolicy::SHARED_UNPINNED,
            NumaSpmvPolicy::SHARED_PINNED,
            NumaSpmvPolicy::REPLICATED_PINNED,
        ] {
            for _ in 0..2 {
                let transpose = numa
                    .try_matvec_transpose_with_policy(&x, policy)
                    .expect("checked transpose policy");
                let hermitian = numa
                    .try_matvec_hermitian_with_policy(&x, policy)
                    .expect("checked Hermitian policy");
                assert_eq!(transpose, expected_transpose);
                assert_eq!(hermitian, expected_hermitian);
            }
        }
    }

    #[test]
    fn checked_numa_matvec_rejects_bad_input_and_partition() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut numa = csr.to_numa_partitioned(1);

        assert!(matches!(
            numa.try_matvec(&[1.0]),
            Err(NumaSpmvError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));

        numa.partitions[0].col_indices[0] = numa.num_cols;
        assert!(matches!(
            numa.try_matvec(&[1.0, 2.0]),
            Err(NumaSpmvError::InvalidPartition { .. })
        ));
    }

    #[test]
    fn checked_numa_linear_operator_rejects_mutated_partition_metadata() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut numa = csr.to_numa_partitioned(1);
        numa.partitions[0].row_ptrs[1] = numa.partitions[0].values.len() + 1;

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| numa.try_apply(&[1.0, 2.0])))
                .expect("checked NUMA operator apply must not panic");
        assert!(matches!(
            result,
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "NUMA CSR SpMV"
        ));
    }

    #[test]
    fn checked_numa_execution_rejects_nonfinite_partition_values() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut numa = csr.to_numa_partitioned(1);
        numa.partitions[0].values[0] = f64::NAN;

        assert!(matches!(
            numa.try_matvec(&[1.0, 2.0]),
            Err(NumaSpmvError::NonFinite {
                operation: "matrix values"
            })
        ));
        assert!(matches!(
            numa.try_apply(&[1.0, 2.0]),
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "NUMA CSR SpMV"
        ));
    }

    #[test]
    fn test_partition_stats_include_placement() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let topology = MachineTopology {
            total_logical_cpus: 2,
            numa_nodes: vec![sonium_solvers_sys::topology::NumaNode {
                id: 7,
                logical_cpus: vec![0, 1],
                memory_bytes: None,
            }],
            sockets: vec![sonium_solvers_sys::topology::SocketInfo {
                id: 3,
                logical_cpus: vec![0, 1],
                numa_node_ids: vec![7],
            }],
            notes: Vec::new(),
        };
        let placement = topology.placement_plan(1);
        let numa = csr.to_numa_partitioned_with_placement(&placement);
        let stats = numa.partition_stats();

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].numa_node_id, Some(7));
        assert_eq!(stats[0].socket_id, Some(3));
        assert_eq!(stats[0].logical_cpu_count, 2);
    }

    #[test]
    fn partitioning_report_records_admission_and_skewed_work_balance() {
        let row_ptrs = vec![0, 1, 2, 3, 13, 14, 15, 16];
        let csr = CsrMatrix::from_raw_parts(
            7,
            7,
            row_ptrs,
            (0..16).map(|index| index % 7).collect(),
            vec![1.0_f64; 16],
        );
        let numa = csr.to_numa_partitioned(3);
        let report = numa.partitioning_report();
        assert_eq!(report.requested_partitions, 3);
        assert_eq!(report.admitted_partitions, 3);
        assert_eq!(report.placement_workers, 3);
        assert_eq!(report.num_rows, 7);
        assert_eq!(report.total_nnz, 16);
        assert!(!report.clamped_to_rows);
        assert!(report.row_imbalance >= 0.0);
        assert!(report.nnz_imbalance <= 1.0);

        let clamped = csr.to_numa_partitioned(99).partitioning_report();
        assert_eq!(clamped.admitted_partitions, 7);
        assert!(clamped.clamped_to_rows);
    }

    #[test]
    fn partitioning_report_marks_missing_affinity_placement() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let placement = PlacementPlan {
            workers: vec![
                sonium_solvers_sys::topology::WorkerPlacement {
                    worker_index: 0,
                    numa_node_id: 0,
                    socket_id: 0,
                    logical_cpus: Vec::new(),
                },
                sonium_solvers_sys::topology::WorkerPlacement {
                    worker_index: 1,
                    numa_node_id: 0,
                    socket_id: 0,
                    logical_cpus: Vec::new(),
                },
            ],
            notes: Vec::new(),
        };
        let report = csr
            .try_to_numa_partitioned_with_placement(&placement)
            .expect("empty affinity lists are a valid fallback placement")
            .partitioning_report();
        assert_eq!(report.admitted_partitions, 2);
        assert_eq!(report.placement_workers, 2);
        assert!(!report.placement_available);
        assert_eq!(report.placement_status, NumaPlacementStatus::Unavailable);
    }

    #[test]
    fn partitioning_report_marks_uma_placement_unavailable() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let topology = MachineTopology {
            total_logical_cpus: 4,
            numa_nodes: vec![sonium_solvers_sys::topology::NumaNode {
                id: 0,
                logical_cpus: vec![0, 1, 2, 3],
                memory_bytes: None,
            }],
            sockets: vec![sonium_solvers_sys::topology::SocketInfo {
                id: 0,
                logical_cpus: vec![0, 1, 2, 3],
                numa_node_ids: vec![0],
            }],
            notes: vec!["synthetic UMA".into()],
        };
        let numa = csr.to_numa_partitioned_with_placement(&topology.placement_plan(2));
        let report = numa.partitioning_report();
        assert!(!report.placement_available);
        assert_eq!(report.placement_status, NumaPlacementStatus::Unavailable);
    }

    #[test]
    fn test_numa_linear_operator() {
        let dense = dense2(
            2,
            2,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let numa = csr.to_numa_partitioned(2);

        assert_eq!(numa.num_rows(), 2);
        assert_eq!(numa.num_cols(), 2);
        assert!(numa.is_square());

        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let y = numa.apply(&x);
        assert_relative_eq!(y[0].re, 6.0, epsilon = 1e-12);
        assert_relative_eq!(y[1].re, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_detect_numa_partitions() {
        let p = detect_numa_partitions();
        assert!(p >= 1);
    }
}
