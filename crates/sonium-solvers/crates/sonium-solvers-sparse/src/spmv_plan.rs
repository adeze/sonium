//! Sparse matrix-vector inspector/executor plans.
//!
//! `SpmvPlan` analyzes a CSR matrix once, records the byte model and row
//! statistics, then owns an execution representation chosen from the available
//! formats. The first Phase 2 slice is deliberately conservative: default
//! `Auto` keeps raw CSR unless reuse and matrix size make compact indices worth
//! trying, or a caller explicitly requests NUMA/compact execution.

use super::{
    BsrMatrix, CsrMatrix, NumaPartitionedCsr,
    csr::{SpmvKernel, detected_spmv_features, selected_spmv_kernel},
    detect_numa_partitions,
    partition::nnz_balanced_ranges,
};
use num_complex::{Complex32, Complex64};
use num_traits::Float;
use rayon::prelude::*;
use sonium_solvers_core::traits::{ComplexField, LinearOperator, SolverError};
use std::any::TypeId;

/// Default parallel-dispatch floors: enough total work (nnz) and enough rows
/// to distribute across threads without per-row overhead dominating.
/// Measured on Apple Silicon; multi-channel x86 servers differ, so both are
/// env-overridable (`SONIUM_SPARSE_PAR_NNZ`, `SONIUM_SPARSE_PAR_ROWS`).
/// Existing tests pin the defaults.
const COMPACT_CSR_PAR_NNZ_THRESHOLD: usize = 50_000;
const COMPACT_CSR_PAR_MIN_ROWS: usize = 4_000;

pub(crate) fn par_nnz_threshold() -> usize {
    // Resolved once per process: SpMV consults this on every call and a
    // `std::env::var` round-trip per call is pure overhead. Same caching
    // contract as `SONIUM_SPARSE_FORCE_SCALAR` in `csr.rs` — set the variable
    // before process start.
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SONIUM_SPARSE_PAR_NNZ")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(COMPACT_CSR_PAR_NNZ_THRESHOLD)
    })
}

pub(crate) fn par_min_rows() -> usize {
    // Same once-per-process caching as `par_nnz_threshold` above.
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SONIUM_SPARSE_PAR_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(COMPACT_CSR_PAR_MIN_ROWS)
    })
}
const SELL_C_SIGMA_PAR_MIN_CHUNKS: usize = 4;
const DEFAULT_SELL_C_SIGMA_CHUNK_SIZE: usize = 16;
const DEFAULT_SELL_C_SIGMA_WINDOW: usize = 256;

fn validate_checked_vector<T: ComplexField>(
    operation: &'static str,
    values: &[T],
    expected: usize,
) -> Result<(), SolverError> {
    if values.len() != expected {
        return Err(SolverError::DimensionMismatch {
            expected,
            got: values.len(),
        });
    }
    validate_checked_finite(operation, values)
}

fn validate_checked_finite<T: ComplexField>(
    operation: &'static str,
    values: &[T],
) -> Result<(), SolverError> {
    if values.iter().any(|value| !value.norm().is_finite()) {
        return Err(SolverError::OperationFailed {
            operation,
            reason: "vector contains NaN or infinity".into(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmvPlanPreference {
    Auto,
    Csr,
    CompactCsrU32,
    NumaCsr,
    RowBucketedCsr,
    SellCSigma,
    Bsr2,
    Bsr3,
    Bsr4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmvPlanFormat {
    Csr,
    CompactCsrU32,
    NumaCsr,
    RowBucketedCsr,
    SellCSigma,
    Bsr2,
    Bsr3,
    Bsr4,
}

#[derive(Debug, Clone)]
pub struct SpmvPlanConfig {
    pub preference: SpmvPlanPreference,
    pub expected_reuse: usize,
    pub prebake_transpose: bool,
    pub prebake_hermitian: bool,
    pub min_reuse_for_compact_indices: usize,
    pub min_saved_index_bytes_for_compact: usize,
    pub min_reuse_for_bsr: usize,
    pub min_saved_bytes_for_bsr: usize,
    pub min_bsr_block_fill_ratio: f64,
    pub min_reuse_for_row_bucket: usize,
    pub min_nnz_for_row_bucket: usize,
    pub min_row_nnz_cv_for_row_bucket: f64,
    pub sell_c_sigma_chunk_size: usize,
    pub sell_c_sigma_window: usize,
    pub min_reuse_for_sell_c_sigma: usize,
    pub min_nnz_for_sell_c_sigma: usize,
    pub min_sell_c_sigma_fill_ratio: f64,
    pub min_nnz_for_numa: usize,
    pub numa_partitions: Option<usize>,
}

impl Default for SpmvPlanConfig {
    fn default() -> Self {
        Self {
            preference: SpmvPlanPreference::Auto,
            expected_reuse: 1,
            prebake_transpose: false,
            prebake_hermitian: false,
            min_reuse_for_compact_indices: 8,
            min_saved_index_bytes_for_compact: 1 << 20,
            min_reuse_for_bsr: 4,
            min_saved_bytes_for_bsr: 1 << 20,
            min_bsr_block_fill_ratio: 0.75,
            min_reuse_for_row_bucket: 4,
            min_nnz_for_row_bucket: 200_000,
            min_row_nnz_cv_for_row_bucket: 1.0,
            sell_c_sigma_chunk_size: DEFAULT_SELL_C_SIGMA_CHUNK_SIZE,
            sell_c_sigma_window: DEFAULT_SELL_C_SIGMA_WINDOW,
            min_reuse_for_sell_c_sigma: 8,
            min_nnz_for_sell_c_sigma: 200_000,
            min_sell_c_sigma_fill_ratio: 0.85,
            min_nnz_for_numa: 2_000_000,
            numa_partitions: None,
        }
    }
}

impl SpmvPlanConfig {
    pub fn validate(&self) -> Result<(), SolverError> {
        if self.numa_partitions == Some(0) {
            return Err(SolverError::OperationFailed {
                operation: "SpMV plan configuration",
                reason: "NUMA partition count must be greater than zero".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SpmvAnalysis {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
    pub empty_rows: usize,
    pub min_row_nnz: usize,
    pub max_row_nnz: usize,
    pub mean_row_nnz: f64,
    pub row_nnz_stddev: f64,
    pub row_nnz_cv: f64,
    pub max_col_index: Option<usize>,
    pub can_use_u32_indices: bool,
    pub csr_usize_bytes_per_apply: u64,
    pub csr_u32_bytes_per_apply: Option<u64>,
    pub saved_index_bytes_per_apply: Option<u64>,
    pub row_bucket: Option<RowBucketPlanAnalysis>,
    pub sell_c_sigma: Option<SellCSigmaPlanAnalysis>,
    pub bsr2: Option<BsrPlanAnalysis>,
    pub bsr3: Option<BsrPlanAnalysis>,
    pub bsr4: Option<BsrPlanAnalysis>,
}

impl SpmvAnalysis {
    pub fn summary(&self) -> String {
        let bsr_summary = [self.bsr2.as_ref(), self.bsr3.as_ref(), self.bsr4.as_ref()]
            .into_iter()
            .flatten()
            .map(|analysis| {
                format!(
                    "bsr{}:fill={:.3},bytes={}",
                    analysis.block_size, analysis.block_fill_ratio, analysis.bytes_per_apply
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let row_bucket_summary = self
            .row_bucket
            .as_ref()
            .map(|analysis| {
                format!(
                    "row_bucket:buckets={},largest={},bytes={}",
                    analysis.bucket_count, analysis.largest_bucket_rows, analysis.bytes_per_apply
                )
            })
            .unwrap_or_default();
        let sell_summary = self
            .sell_c_sigma
            .as_ref()
            .map(|analysis| {
                format!(
                    "sell_c_sigma:C={},sigma={},fill={:.3},padded_nnz={},bytes={}",
                    analysis.chunk_size,
                    analysis.sigma,
                    analysis.fill_ratio,
                    analysis.padded_nnz,
                    analysis.bytes_per_apply
                )
            })
            .unwrap_or_default();
        format!(
            "rows={} cols={} nnz={} row_nnz=min:{} mean:{:.2} max:{} stddev:{:.2} cv:{:.3} empty={} u32_indices={} bytes/apply usize={} u32={:?} {} {} {}",
            self.nrows,
            self.ncols,
            self.nnz,
            self.min_row_nnz,
            self.mean_row_nnz,
            self.max_row_nnz,
            self.row_nnz_stddev,
            self.row_nnz_cv,
            self.empty_rows,
            self.can_use_u32_indices,
            self.csr_usize_bytes_per_apply,
            self.csr_u32_bytes_per_apply,
            row_bucket_summary,
            sell_summary,
            bsr_summary
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowBucketPlanAnalysis {
    pub bucket_count: usize,
    pub largest_bucket_rows: usize,
    pub singleton_buckets: usize,
    pub bytes_per_apply: u64,
    pub row_order_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SellCSigmaPlanAnalysis {
    pub chunk_size: usize,
    pub sigma: usize,
    pub chunks: usize,
    pub padded_nnz: usize,
    pub fill_ratio: f64,
    pub bytes_per_apply: u64,
    pub padding_overhead_nnz: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsrPlanAnalysis {
    pub block_size: usize,
    pub block_rows: usize,
    pub block_cols: usize,
    pub nnz_blocks: usize,
    pub padded_scalar_nnz: usize,
    pub block_fill_ratio: f64,
    pub bytes_per_apply: u64,
    pub saved_bytes_per_apply: u64,
}

#[derive(Debug, Clone)]
pub struct SpmvPlanReport {
    pub format: SpmvPlanFormat,
    /// Forward-action kernel selected for the concrete scalar type and plan.
    pub selected_kernel: SpmvKernel,
    /// Runtime CPU/build feature state used for dispatch evidence.
    pub features: String,
    pub analysis: SpmvAnalysis,
    pub expected_reuse: usize,
    pub chosen_index_bytes: usize,
    pub chosen_bytes_per_apply: u64,
    pub estimated_saved_bytes_per_apply: u64,
    pub estimated_saved_bytes_over_reuse: u64,
    pub transpose_prebaked: bool,
    pub hermitian_prebaked: bool,
    pub reasons: Vec<String>,
}

impl SpmvPlanReport {
    pub fn summary(&self) -> String {
        format!(
            "format={:?}; selected_kernel={}; features={}; chosen_index_bytes={}; chosen_bytes/apply={}; saved_bytes/apply={}; expected_reuse={}; saved_bytes/reuse={}; transpose_prebaked={}; hermitian_prebaked={}; {}; reasons={}",
            self.format,
            self.selected_kernel.name(),
            self.features,
            self.chosen_index_bytes,
            self.chosen_bytes_per_apply,
            self.estimated_saved_bytes_per_apply,
            self.expected_reuse,
            self.estimated_saved_bytes_over_reuse,
            self.transpose_prebaked,
            self.hermitian_prebaked,
            self.analysis.summary(),
            self.reasons.join("; ")
        )
    }
}

fn selected_plan_kernel<T: ComplexField>(format: SpmvPlanFormat) -> SpmvKernel {
    match format {
        SpmvPlanFormat::Csr | SpmvPlanFormat::CompactCsrU32 | SpmvPlanFormat::SellCSigma => {
            selected_spmv_kernel::<T>()
        }
        // Row bucketing and BSR currently use their scalar row/block kernels.
        SpmvPlanFormat::RowBucketedCsr
        | SpmvPlanFormat::Bsr2
        | SpmvPlanFormat::Bsr3
        | SpmvPlanFormat::Bsr4 => SpmvKernel::Scalar,
        // NUMA has a specialized f64 forward path; other scalar families use
        // its generic partition loop.
        SpmvPlanFormat::NumaCsr => {
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                selected_spmv_kernel::<f64>()
            } else {
                SpmvKernel::Scalar
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum SpmvPlan<T: ComplexField> {
    Csr {
        matrix: CsrMatrix<T>,
        transpose: Option<CompactCsrMatrix<T>>,
        hermitian: Option<CompactCsrMatrix<T>>,
        report: SpmvPlanReport,
    },
    CompactCsrU32 {
        matrix: CompactCsrMatrix<T>,
        transpose: Option<CompactCsrMatrix<T>>,
        hermitian: Option<CompactCsrMatrix<T>>,
        report: SpmvPlanReport,
    },
    NumaCsr {
        matrix: NumaPartitionedCsr<T>,
        transpose: Option<CompactCsrMatrix<T>>,
        hermitian: Option<CompactCsrMatrix<T>>,
        report: SpmvPlanReport,
    },
    RowBucketedCsr {
        matrix: RowBucketedCsrMatrix<T>,
        transpose: Option<CompactCsrMatrix<T>>,
        hermitian: Option<CompactCsrMatrix<T>>,
        report: SpmvPlanReport,
    },
    SellCSigma {
        matrix: SellCSigmaMatrix<T>,
        transpose: Option<CompactCsrMatrix<T>>,
        hermitian: Option<CompactCsrMatrix<T>>,
        report: SpmvPlanReport,
    },
    Bsr2 {
        matrix: BsrMatrix<T, 2>,
        report: SpmvPlanReport,
    },
    Bsr3 {
        matrix: BsrMatrix<T, 3>,
        report: SpmvPlanReport,
    },
    Bsr4 {
        matrix: BsrMatrix<T, 4>,
        report: SpmvPlanReport,
    },
}

impl<T: ComplexField> SpmvPlan<T> {
    /// Build an SpMV plan after validating the mutable CSR storage.
    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
        config: SpmvPlanConfig,
    ) -> Result<Self, SolverError> {
        config.validate()?;
        matrix.validate_structure()?;
        Self::from_csr_unchecked(matrix, config)
    }

    /// Build an SpMV plan, retaining the historical panic-on-invalid API.
    pub fn from_csr(matrix: &CsrMatrix<T>, config: SpmvPlanConfig) -> Self {
        Self::try_from_csr(matrix, config)
            .unwrap_or_else(|error| panic!("invalid CSR matrix for SpMV plan: {error}"))
    }

    fn from_csr_unchecked(
        matrix: &CsrMatrix<T>,
        config: SpmvPlanConfig,
    ) -> Result<Self, SolverError> {
        let analysis = analyze_csr(matrix, &config);
        let (format, mut reasons) = choose_format(&analysis, &config);
        let can_prebake_transpose = matches!(
            format,
            SpmvPlanFormat::Csr
                | SpmvPlanFormat::CompactCsrU32
                | SpmvPlanFormat::NumaCsr
                | SpmvPlanFormat::RowBucketedCsr
                | SpmvPlanFormat::SellCSigma
        );
        let transpose = if config.prebake_transpose && can_prebake_transpose {
            if indices_fit_u32(matrix) {
                Some(CompactCsrMatrix::try_transpose_from_csr(matrix)?)
            } else {
                None
            }
        } else {
            None
        };
        let hermitian = if config.prebake_hermitian && can_prebake_transpose {
            if indices_fit_u32(matrix) {
                Some(CompactCsrMatrix::try_hermitian_from_csr(matrix)?)
            } else {
                None
            }
        } else {
            None
        };
        if config.prebake_transpose {
            if transpose.is_some() {
                reasons.push("prebaked transpose CSR with u32 indices".to_string());
            } else if matches!(
                format,
                SpmvPlanFormat::Bsr2 | SpmvPlanFormat::Bsr3 | SpmvPlanFormat::Bsr4
            ) {
                reasons
                    .push("transpose prebake skipped: BSR applies transpose directly".to_string());
            } else {
                reasons.push("transpose prebake requested, but indices do not fit u32".to_string());
            }
        }
        if config.prebake_hermitian {
            if hermitian.is_some() {
                reasons.push("prebaked Hermitian CSR with u32 indices".to_string());
            } else if matches!(
                format,
                SpmvPlanFormat::Bsr2 | SpmvPlanFormat::Bsr3 | SpmvPlanFormat::Bsr4
            ) {
                reasons
                    .push("Hermitian prebake skipped: BSR applies Hermitian directly".to_string());
            } else {
                reasons.push("Hermitian prebake requested, but indices do not fit u32".to_string());
            }
        }
        let chosen_bytes_per_apply = chosen_bytes_per_apply(format, &analysis);
        let estimated_saved_bytes_per_apply = analysis
            .csr_usize_bytes_per_apply
            .saturating_sub(chosen_bytes_per_apply);
        let report = SpmvPlanReport {
            format,
            selected_kernel: selected_plan_kernel::<T>(format),
            features: detected_spmv_features(),
            analysis,
            expected_reuse: config.expected_reuse,
            chosen_index_bytes: chosen_index_bytes(format),
            chosen_bytes_per_apply,
            estimated_saved_bytes_per_apply,
            estimated_saved_bytes_over_reuse: estimated_saved_bytes_per_apply
                .saturating_mul(config.expected_reuse as u64),
            transpose_prebaked: transpose.is_some(),
            hermitian_prebaked: hermitian.is_some(),
            reasons,
        };

        match format {
            SpmvPlanFormat::Csr => Ok(Self::Csr {
                matrix: matrix.clone(),
                transpose,
                hermitian,
                report,
            }),
            SpmvPlanFormat::CompactCsrU32 => Ok(Self::CompactCsrU32 {
                matrix: CompactCsrMatrix::try_from_csr(matrix)?,
                transpose,
                hermitian,
                report,
            }),
            SpmvPlanFormat::NumaCsr => {
                let partitions = config
                    .numa_partitions
                    .unwrap_or_else(detect_numa_partitions);
                Ok(Self::NumaCsr {
                    matrix: NumaPartitionedCsr::try_from_csr(matrix, partitions)?,
                    transpose,
                    hermitian,
                    report,
                })
            }
            SpmvPlanFormat::RowBucketedCsr => Ok(Self::RowBucketedCsr {
                matrix: RowBucketedCsrMatrix::try_from_csr(matrix)?,
                transpose,
                hermitian,
                report,
            }),
            SpmvPlanFormat::SellCSigma => Ok(Self::SellCSigma {
                matrix: SellCSigmaMatrix::try_from_csr(
                    matrix,
                    config.sell_c_sigma_chunk_size,
                    config.sell_c_sigma_window,
                )?,
                transpose,
                hermitian,
                report,
            }),
            SpmvPlanFormat::Bsr2 => Ok(Self::Bsr2 {
                matrix: BsrMatrix::try_from_csr(matrix)?,
                report,
            }),
            SpmvPlanFormat::Bsr3 => Ok(Self::Bsr3 {
                matrix: BsrMatrix::try_from_csr(matrix)?,
                report,
            }),
            SpmvPlanFormat::Bsr4 => Ok(Self::Bsr4 {
                matrix: BsrMatrix::try_from_csr(matrix)?,
                report,
            }),
        }
    }

    pub fn auto(matrix: &CsrMatrix<T>) -> Self {
        Self::from_csr(matrix, SpmvPlanConfig::default())
    }

    pub fn report(&self) -> &SpmvPlanReport {
        match self {
            Self::Csr { report, .. } => report,
            Self::CompactCsrU32 { report, .. } => report,
            Self::NumaCsr { report, .. } => report,
            Self::RowBucketedCsr { report, .. } => report,
            Self::SellCSigma { report, .. } => report,
            Self::Bsr2 { report, .. } => report,
            Self::Bsr3 { report, .. } => report,
            Self::Bsr4 { report, .. } => report,
        }
    }

    pub fn format(&self) -> SpmvPlanFormat {
        self.report().format
    }

    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        match self {
            Self::Csr { matrix, .. } => matrix.matvec(x),
            Self::CompactCsrU32 { matrix, .. } => matrix.matvec(x),
            Self::NumaCsr { matrix, .. } => matrix.matvec(x),
            Self::RowBucketedCsr { matrix, .. } => matrix.matvec(x),
            Self::SellCSigma { matrix, .. } => matrix.matvec(x),
            Self::Bsr2 { matrix, .. } => matrix.apply(x),
            Self::Bsr3 { matrix, .. } => matrix.apply(x),
            Self::Bsr4 { matrix, .. } => matrix.apply(x),
        }
    }
}

impl<T: ComplexField> LinearOperator<T> for SpmvPlan<T> {
    fn num_rows(&self) -> usize {
        self.report().analysis.nrows
    }

    fn num_cols(&self) -> usize {
        self.report().analysis.ncols
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        self.matvec(x)
    }

    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        match self {
            Self::Csr { matrix, .. } => matrix.try_apply(x),
            Self::CompactCsrU32 { matrix, .. } => matrix.try_apply(x),
            Self::NumaCsr { matrix, .. } => matrix.try_apply(x),
            Self::RowBucketedCsr { matrix, .. } => matrix.try_apply(x),
            Self::SellCSigma { matrix, .. } => matrix.try_apply(x),
            Self::Bsr2 { matrix, .. } => matrix.try_apply(x),
            Self::Bsr3 { matrix, .. } => matrix.try_apply(x),
            Self::Bsr4 { matrix, .. } => matrix.try_apply(x),
        }
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Csr { matrix, .. } => matrix.matvec_into_buf(x, y),
            Self::CompactCsrU32 { matrix, .. } => matrix.matvec_into_buf(x, y),
            Self::NumaCsr { matrix, .. } => matrix.apply_into(x, y),
            Self::RowBucketedCsr { matrix, .. } => matrix.matvec_into_buf(x, y),
            Self::SellCSigma { matrix, .. } => matrix.matvec_into_buf(x, y),
            Self::Bsr2 { matrix, .. } => matrix.apply_into(x, y),
            Self::Bsr3 { matrix, .. } => matrix.apply_into(x, y),
            Self::Bsr4 { matrix, .. } => matrix.apply_into(x, y),
        }
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Csr {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.matvec_into_buf(x, y),
                None => matrix.matvec_transpose_into_buf(x, y),
            },
            Self::CompactCsrU32 {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.matvec_into_buf(x, y),
                None => matrix.matvec_transpose_into_buf(x, y),
            },
            Self::NumaCsr {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.matvec_into_buf(x, y),
                None => matrix.apply_transpose_into(x, y),
            },
            Self::RowBucketedCsr {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.matvec_into_buf(x, y),
                None => matrix.matvec_transpose_into_buf(x, y),
            },
            Self::SellCSigma {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.matvec_into_buf(x, y),
                None => matrix.apply_transpose_into(x, y),
            },
            Self::Bsr2 { matrix, .. } => matrix.apply_transpose_into(x, y),
            Self::Bsr3 { matrix, .. } => matrix.apply_transpose_into(x, y),
            Self::Bsr4 { matrix, .. } => matrix.apply_transpose_into(x, y),
        }
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        match self {
            Self::Csr {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.try_apply(x),
                None => matrix.try_apply_transpose(x),
            },
            Self::CompactCsrU32 {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.try_apply(x),
                None => matrix.try_apply_transpose(x),
            },
            Self::NumaCsr {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.try_apply(x),
                None => matrix.try_apply_transpose(x),
            },
            Self::RowBucketedCsr {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.try_apply(x),
                None => matrix.try_apply_transpose(x),
            },
            Self::SellCSigma {
                matrix, transpose, ..
            } => match transpose {
                Some(transpose) => transpose.try_apply(x),
                None => matrix.try_apply_transpose(x),
            },
            Self::Bsr2 { matrix, .. } => matrix.try_apply_transpose(x),
            Self::Bsr3 { matrix, .. } => matrix.try_apply_transpose(x),
            Self::Bsr4 { matrix, .. } => matrix.try_apply_transpose(x),
        }
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        match self {
            Self::Csr {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.matvec(x),
                None => matrix.matvec_hermitian(x),
            },
            Self::CompactCsrU32 {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.matvec(x),
                None => matrix.matvec_hermitian(x),
            },
            Self::NumaCsr {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.matvec(x),
                None => matrix.matvec_hermitian(x),
            },
            Self::RowBucketedCsr {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.matvec(x),
                None => matrix.matvec_hermitian(x),
            },
            Self::SellCSigma {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.matvec(x),
                None => matrix.apply_hermitian(x),
            },
            Self::Bsr2 { matrix, .. } => matrix.apply_hermitian(x),
            Self::Bsr3 { matrix, .. } => matrix.apply_hermitian(x),
            Self::Bsr4 { matrix, .. } => matrix.apply_hermitian(x),
        }
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        match self {
            Self::Csr {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.try_apply(x),
                None => matrix.try_apply_hermitian(x),
            },
            Self::CompactCsrU32 {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.try_apply(x),
                None => matrix.try_apply_hermitian(x),
            },
            Self::NumaCsr {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.try_apply(x),
                None => matrix.try_apply_hermitian(x),
            },
            Self::RowBucketedCsr {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.try_apply(x),
                None => matrix.try_apply_hermitian(x),
            },
            Self::SellCSigma {
                matrix, hermitian, ..
            } => match hermitian {
                Some(hermitian) => hermitian.try_apply(x),
                None => matrix.try_apply_hermitian(x),
            },
            Self::Bsr2 { matrix, .. } => matrix.try_apply_hermitian(x),
            Self::Bsr3 { matrix, .. } => matrix.try_apply_hermitian(x),
            Self::Bsr4 { matrix, .. } => matrix.try_apply_hermitian(x),
        }
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }
}

#[derive(Debug, Clone)]
pub struct CompactCsrMatrix<T: ComplexField> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_ptrs: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<T>,
}

impl<T: ComplexField> CompactCsrMatrix<T> {
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Option<Self> {
        Self::try_from_csr(matrix).ok()
    }

    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        if !indices_fit_u32(matrix) {
            return Err(SolverError::OperationFailed {
                operation: "compact CSR construction",
                reason: "row-pointer or column index does not fit u32".into(),
            });
        }

        Ok(Self {
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            row_ptrs: matrix.row_ptrs.iter().map(|&v| v as u32).collect(),
            col_indices: matrix.col_indices.iter().map(|&v| v as u32).collect(),
            values: matrix.values.clone(),
        })
    }

    pub fn transpose_from_csr(matrix: &CsrMatrix<T>) -> Option<Self> {
        Self::try_transpose_from_csr(matrix).ok()
    }

    pub fn try_transpose_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        if !indices_fit_u32(matrix) {
            return Err(SolverError::OperationFailed {
                operation: "compact CSR transpose construction",
                reason: "row-pointer or column index does not fit u32".into(),
            });
        }

        Ok(Self::transpose_from_csr_with_value_map(matrix, |value| {
            value
        }))
    }

    pub fn hermitian_from_csr(matrix: &CsrMatrix<T>) -> Option<Self> {
        Self::try_hermitian_from_csr(matrix).ok()
    }

    pub fn try_hermitian_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        if !indices_fit_u32(matrix) {
            return Err(SolverError::OperationFailed {
                operation: "compact CSR Hermitian construction",
                reason: "row-pointer or column index does not fit u32".into(),
            });
        }

        Ok(Self::transpose_from_csr_with_value_map(matrix, |value| {
            value.conj()
        }))
    }

    fn transpose_from_csr_with_value_map<F>(matrix: &CsrMatrix<T>, mut map_value: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        let mut row_counts = vec![0usize; matrix.num_cols];
        for &col in &matrix.col_indices {
            row_counts[col] += 1;
        }

        let mut row_ptrs_usize = vec![0usize; matrix.num_cols + 1];
        for row in 0..matrix.num_cols {
            row_ptrs_usize[row + 1] = row_ptrs_usize[row] + row_counts[row];
        }

        let nnz = matrix.nnz();
        let mut values = vec![T::zero(); nnz];
        let mut col_indices = vec![0u32; nnz];
        let mut next = row_ptrs_usize.clone();

        for row in 0..matrix.num_rows {
            for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let col = matrix.col_indices[idx];
                let dst = next[col];
                values[dst] = map_value(matrix.values[idx]);
                col_indices[dst] = row as u32;
                next[col] += 1;
            }
        }

        Self {
            num_rows: matrix.num_cols,
            num_cols: matrix.num_rows,
            row_ptrs: row_ptrs_usize.into_iter().map(|v| v as u32).collect(),
            col_indices,
            values,
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        let mut y = vec![T::zero(); self.num_rows];
        self.matvec_into_buf(x, &mut y);
        y
    }

    /// Matrix-vector product into a pre-allocated output slice.
    ///
    /// Dispatches to SIMD gather-dot kernels (AVX2/NEON) for the four
    /// concrete scalar types, mirroring `CsrMatrix::matvec_into_buf`. Uses
    /// the same parallelization heuristic as `CsrMatrix`: rayon row-parallel
    /// execution when both the nonzero count and the row count are large
    /// enough to amortize thread overhead.
    pub fn matvec_into_buf(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_rows, "Output vector size mismatch");

        let nnz = self.nnz();
        if nnz >= par_nnz_threshold()
            && self.num_rows >= par_min_rows()
            && sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(nnz) > 1
        {
            self.matvec_parallel_into(x, y);
            return;
        }

        self.matvec_sequential_into(x, y);
    }

    /// Checked variant of [`Self::matvec`] that revalidates the public
    /// compact-CSR storage before any indexed or SIMD access. The compact
    /// fields remain public for compatibility, so construction-time checks
    /// alone cannot protect a matrix that has been mutated afterwards.
    pub fn try_matvec(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_structure("compact CSR SpMV")?;
        self.validate_checked_input("compact CSR SpMV", x, self.num_cols)?;
        let y = self.matvec(x);
        self.validate_checked_output("compact CSR SpMV", &y, self.num_rows)?;
        Ok(y)
    }

    /// Checked variant of [`Self::matvec_into_buf`].
    pub fn try_matvec_into_buf(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_structure("compact CSR SpMV")?;
        self.validate_checked_input("compact CSR SpMV", x, self.num_cols)?;
        self.validate_checked_output("compact CSR SpMV", y, self.num_rows)?;
        let mut staged = vec![T::zero(); self.num_rows];
        self.matvec_into_buf(x, &mut staged);
        self.validate_finite_output("compact CSR SpMV", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    /// Sequential matvec with per-type SIMD dispatch.
    ///
    /// The AVX2 gathers reinterpret u32 column indices as i32, so the f64/f32
    /// SIMD paths are guarded by `num_cols <= i32::MAX` (larger matrices fall
    /// back to the generic scalar loop). The complex kernels read indices as
    /// scalars and cover the full u32 range.
    #[inline]
    fn matvec_sequential_into(&self, x: &[T], y: &mut [T]) {
        if TypeId::of::<T>() == TypeId::of::<f64>() && self.num_cols <= i32::MAX as usize {
            // SAFETY: We just verified T == f64. The pointer casts are
            // transmuting &[T] <-> &[f64] which is layout-identical.
            unsafe {
                let x_f64 = &*(x as *const [T] as *const [f64]);
                let y_f64 = &mut *(y as *mut [T] as *mut [f64]);
                let values_f64 = &*(self.values.as_slice() as *const [T] as *const [f64]);
                compact_spmv_f64_simd(
                    values_f64,
                    self.col_indices.as_slice(),
                    self.row_ptrs.as_slice(),
                    x_f64,
                    y_f64,
                    self.num_rows,
                    true,
                );
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            // SAFETY: We just verified T == Complex64. Complex64 is
            // `Complex<f64>` with layout `(f64, f64)`, so the cast is valid.
            unsafe {
                let x_c64 = &*(x as *const [T] as *const [Complex64]);
                let y_c64 = &mut *(y as *mut [T] as *mut [Complex64]);
                let values_c64 = &*(self.values.as_slice() as *const [T] as *const [Complex64]);
                compact_spmv_complex64_simd(
                    values_c64,
                    self.col_indices.as_slice(),
                    self.row_ptrs.as_slice(),
                    x_c64,
                    y_c64,
                    self.num_rows,
                );
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() && self.num_cols <= i32::MAX as usize {
            // SAFETY: We just verified T == f32. The pointer cast is valid.
            unsafe {
                let x_f32 = &*(x as *const [T] as *const [f32]);
                let y_f32 = &mut *(y as *mut [T] as *mut [f32]);
                let values_f32 = &*(self.values.as_slice() as *const [T] as *const [f32]);
                compact_spmv_f32_simd(
                    values_f32,
                    self.col_indices.as_slice(),
                    self.row_ptrs.as_slice(),
                    x_f32,
                    y_f32,
                    self.num_rows,
                    true,
                );
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            // SAFETY: We just verified T == Complex32. Complex32 is
            // `Complex<f32>` with layout `(f32, f32)`, so the cast is valid.
            unsafe {
                let x_c32 = &*(x as *const [T] as *const [Complex32]);
                let y_c32 = &mut *(y as *mut [T] as *mut [Complex32]);
                let values_c32 = &*(self.values.as_slice() as *const [T] as *const [Complex32]);
                compact_spmv_complex32_simd(
                    values_c32,
                    self.col_indices.as_slice(),
                    self.row_ptrs.as_slice(),
                    x_c32,
                    y_c32,
                    self.num_rows,
                );
            }
            return;
        }

        // Generic scalar fallback for all other types.
        for (row, yi) in y.iter_mut().enumerate().take(self.num_rows) {
            let start = self.row_ptrs[row] as usize;
            let end = self.row_ptrs[row + 1] as usize;
            let mut sum = T::zero();
            for idx in start..end {
                let col = self.col_indices[idx] as usize;
                sum += self.values[idx] * x[col];
            }
            *yi = sum;
        }
    }

    /// Parallel matvec: rows are split into contiguous ranges with ≈ equal
    /// nonzero counts (one rayon task per range, each running the SIMD
    /// per-row kernel for the concrete scalar type), so skewed matrices do
    /// not leave a straggler thread. Runs on the bandwidth-limited pool
    /// chosen by `runtime::bandwidth_threads_for_nnz`.
    fn matvec_parallel_into(&self, x: &[T], y: &mut [T]) {
        let values = &self.values;
        let col_indices = &self.col_indices;
        let row_ptrs = &self.row_ptrs;
        let nnz = self.nnz();

        sonium_solvers_sys::runtime::with_bandwidth_pool_for_nnz(nnz, || {
            let nthreads = rayon::current_num_threads().max(1);
            let ranges = nnz_balanced_ranges(row_ptrs.as_slice(), nthreads);

            // Split `y` into one contiguous chunk per nnz-balanced row range
            // so each rayon task writes a disjoint slice of the output.
            let mut y_rest = &mut *y;
            let mut prev_end = 0usize;
            let mut y_chunks: Vec<&mut [T]> = Vec::with_capacity(ranges.len());
            for range in &ranges {
                let (head, tail) = y_rest.split_at_mut(range.end - prev_end);
                prev_end = range.end;
                y_chunks.push(head);
                y_rest = tail;
            }

            if TypeId::of::<T>() == TypeId::of::<f64>() && self.num_cols <= i32::MAX as usize {
                // SAFETY: We verified T == f64. Each chunk is a disjoint
                // &mut [f64], so there are no data races between tasks.
                let values_f64 = unsafe { &*(values.as_slice() as *const [T] as *const [f64]) };
                let x_f64 = unsafe { &*(x as *const [T] as *const [f64]) };
                let y_chunks_f64: Vec<&mut [f64]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [f64]) })
                    .collect();

                y_chunks_f64
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i] as usize;
                            let end = row_ptrs[i + 1] as usize;
                            // SAFETY: CSR invariants guarantee valid indices.
                            *yi = unsafe {
                                compact_spmv_row_f64(
                                    values_f64,
                                    col_indices,
                                    x_f64,
                                    start,
                                    end,
                                    true,
                                )
                            };
                        }
                    });
                return;
            }

            if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                let values_c64 =
                    unsafe { &*(values.as_slice() as *const [T] as *const [Complex64]) };
                let x_c64 = unsafe { &*(x as *const [T] as *const [Complex64]) };
                let y_chunks_c64: Vec<&mut [Complex64]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [Complex64]) })
                    .collect();

                y_chunks_c64
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i] as usize;
                            let end = row_ptrs[i + 1] as usize;
                            *yi = unsafe {
                                compact_spmv_row_complex64(
                                    values_c64,
                                    col_indices,
                                    x_c64,
                                    start,
                                    end,
                                )
                            };
                        }
                    });
                return;
            }

            if TypeId::of::<T>() == TypeId::of::<f32>() && self.num_cols <= i32::MAX as usize {
                let values_f32 = unsafe { &*(values.as_slice() as *const [T] as *const [f32]) };
                let x_f32 = unsafe { &*(x as *const [T] as *const [f32]) };
                let y_chunks_f32: Vec<&mut [f32]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [f32]) })
                    .collect();

                y_chunks_f32
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i] as usize;
                            let end = row_ptrs[i + 1] as usize;
                            *yi = unsafe {
                                compact_spmv_row_f32(
                                    values_f32,
                                    col_indices,
                                    x_f32,
                                    start,
                                    end,
                                    true,
                                )
                            };
                        }
                    });
                return;
            }

            if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                let values_c32 =
                    unsafe { &*(values.as_slice() as *const [T] as *const [Complex32]) };
                let x_c32 = unsafe { &*(x as *const [T] as *const [Complex32]) };
                let y_chunks_c32: Vec<&mut [Complex32]> = y_chunks
                    .into_iter()
                    .map(|chunk| unsafe { &mut *(chunk as *mut [T] as *mut [Complex32]) })
                    .collect();

                y_chunks_c32
                    .into_par_iter()
                    .zip(ranges.par_iter())
                    .for_each(|(chunk, range)| {
                        for (offset, yi) in chunk.iter_mut().enumerate() {
                            let i = range.start + offset;
                            let start = row_ptrs[i] as usize;
                            let end = row_ptrs[i + 1] as usize;
                            *yi = unsafe {
                                compact_spmv_row_complex32(
                                    values_c32,
                                    col_indices,
                                    x_c32,
                                    start,
                                    end,
                                )
                            };
                        }
                    });
                return;
            }

            // Generic scalar fallback for all other types.
            y_chunks
                .into_par_iter()
                .zip(ranges.par_iter())
                .for_each(|(chunk, range)| {
                    for (offset, yi) in chunk.iter_mut().enumerate() {
                        let i = range.start + offset;
                        let start = row_ptrs[i] as usize;
                        let end = row_ptrs[i + 1] as usize;
                        let mut sum = T::zero();
                        for idx in start..end {
                            let col = col_indices[idx] as usize;
                            sum += values[idx] * x[col];
                        }
                        *yi = sum;
                    }
                });
        });
    }

    pub fn matvec_transpose(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        let mut y = vec![T::zero(); self.num_cols];
        self.matvec_transpose_into_buf(x, &mut y);
        y
    }

    pub fn matvec_transpose_into_buf(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_cols, "Output vector size mismatch");
        y.fill(T::zero());

        for (row, &xi) in x.iter().enumerate() {
            let start = self.row_ptrs[row] as usize;
            let end = self.row_ptrs[row + 1] as usize;
            for idx in start..end {
                let col = self.col_indices[idx] as usize;
                y[col] += self.values[idx] * xi;
            }
        }
    }

    /// Checked variant of [`Self::matvec_transpose`].
    pub fn try_matvec_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_structure("compact CSR transpose SpMV")?;
        self.validate_checked_input("compact CSR transpose SpMV", x, self.num_rows)?;
        let y = self.matvec_transpose(x);
        self.validate_checked_output("compact CSR transpose SpMV", &y, self.num_cols)?;
        Ok(y)
    }

    /// Checked variant of [`Self::matvec_transpose_into_buf`].
    pub fn try_matvec_transpose_into_buf(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_structure("compact CSR transpose SpMV")?;
        self.validate_checked_input("compact CSR transpose SpMV", x, self.num_rows)?;
        self.validate_checked_output("compact CSR transpose SpMV", y, self.num_cols)?;
        let mut staged = vec![T::zero(); self.num_cols];
        self.matvec_transpose_into_buf(x, &mut staged);
        self.validate_finite_output("compact CSR transpose SpMV", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    pub fn matvec_hermitian(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        let mut y = vec![T::zero(); self.num_cols];
        self.matvec_hermitian_into_buf(x, &mut y);
        y
    }

    pub fn matvec_hermitian_into_buf(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_rows, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_cols, "Output vector size mismatch");
        y.fill(T::zero());

        for (row, &xi) in x.iter().enumerate() {
            let start = self.row_ptrs[row] as usize;
            let end = self.row_ptrs[row + 1] as usize;
            for idx in start..end {
                let col = self.col_indices[idx] as usize;
                y[col] += self.values[idx].conj() * xi;
            }
        }
    }

    /// Checked variant of [`Self::matvec_hermitian`].
    pub fn try_matvec_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_structure("compact CSR Hermitian SpMV")?;
        self.validate_checked_input("compact CSR Hermitian SpMV", x, self.num_rows)?;
        let y = self.matvec_hermitian(x);
        self.validate_checked_output("compact CSR Hermitian SpMV", &y, self.num_cols)?;
        Ok(y)
    }

    /// Checked variant of [`Self::matvec_hermitian_into_buf`].
    pub fn try_matvec_hermitian_into_buf(&self, x: &[T], y: &mut [T]) -> Result<(), SolverError> {
        self.validate_checked_structure("compact CSR Hermitian SpMV")?;
        self.validate_checked_input("compact CSR Hermitian SpMV", x, self.num_rows)?;
        self.validate_checked_output("compact CSR Hermitian SpMV", y, self.num_cols)?;
        let mut staged = vec![T::zero(); self.num_cols];
        self.matvec_hermitian_into_buf(x, &mut staged);
        self.validate_finite_output("compact CSR Hermitian SpMV", &staged)?;
        y.copy_from_slice(&staged);
        Ok(())
    }

    fn validate_checked_structure(&self, operation: &'static str) -> Result<(), SolverError> {
        if self.row_ptrs.len() != self.num_rows.saturating_add(1) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "row-pointer length {} != num_rows + 1 ({})",
                    self.row_ptrs.len(),
                    self.num_rows.saturating_add(1)
                ),
            });
        }
        if self.row_ptrs.first().copied() != Some(0) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "row pointers must start at zero".into(),
            });
        }
        if self.col_indices.len() != self.values.len() {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "column-index length {} != value length {}",
                    self.col_indices.len(),
                    self.values.len()
                ),
            });
        }
        let mut previous = 0usize;
        for (row, &end) in self.row_ptrs.iter().enumerate().skip(1) {
            let end = end as usize;
            if end < previous || end > self.values.len() {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("invalid row pointer at row {row}: {end}"),
                });
            }
            previous = end;
        }
        if previous != self.values.len() {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "terminal row pointer {previous} != nnz {}",
                    self.values.len()
                ),
            });
        }
        if self
            .col_indices
            .iter()
            .any(|&column| column as usize >= self.num_cols)
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "column index is outside matrix dimensions".into(),
            });
        }
        if self.values.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "matrix values contain NaN or infinity".into(),
            });
        }
        Ok(())
    }

    fn validate_checked_input(
        &self,
        operation: &'static str,
        x: &[T],
        expected: usize,
    ) -> Result<(), SolverError> {
        if x.len() != expected {
            return Err(SolverError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        if x.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "input vector contains NaN or infinity".into(),
            });
        }
        Ok(())
    }

    fn validate_checked_output(
        &self,
        operation: &'static str,
        y: &[T],
        expected: usize,
    ) -> Result<(), SolverError> {
        if y.len() != expected {
            return Err(SolverError::DimensionMismatch {
                expected,
                got: y.len(),
            });
        }
        self.validate_finite_output(operation, y)
    }

    fn validate_finite_output(&self, operation: &'static str, y: &[T]) -> Result<(), SolverError> {
        if y.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "output vector contains NaN or infinity".into(),
            });
        }
        Ok(())
    }
}

impl<T: ComplexField> LinearOperator<T> for CompactCsrMatrix<T> {
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
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_into_buf(x, y);
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_transpose_into_buf(x, y);
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_transpose(x)
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        self.matvec_hermitian(x)
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec_hermitian(x)
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }
}

/// CSR reordered so rows of equal length are contiguous ("row bucketing").
///
/// Historically this existed because the plain-CSR parallel SpMV split rows
/// equally across threads, leaving a straggler thread on skewed matrices.
/// The plain CSR and compact-CSR parallel paths now partition rows by
/// nonzero count (see
/// [`partition::nnz_balanced_ranges`](crate::partition::nnz_balanced_ranges)),
/// so this format's remaining edge is limited to extremely skewed matrices
/// (`row_nnz_cv >= 1`) with high apply reuse, where processing buckets of
/// uniform-length rows keeps every per-row kernel equally efficient.
#[derive(Debug, Clone)]
pub struct RowBucketedCsrMatrix<T: ComplexField> {
    pub matrix: CsrMatrix<T>,
    pub row_order: Vec<usize>,
    pub bucket_offsets: Vec<usize>,
}

impl<T: ComplexField> RowBucketedCsrMatrix<T> {
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Self {
        Self::try_from_csr(matrix)
            .unwrap_or_else(|error| panic!("invalid CSR matrix for row bucketing: {error}"))
    }

    pub fn try_from_csr(matrix: &CsrMatrix<T>) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        Ok(Self::from_csr_unchecked(matrix))
    }

    fn from_csr_unchecked(matrix: &CsrMatrix<T>) -> Self {
        let mut rows = (0..matrix.num_rows)
            .map(|row| {
                let row_nnz = matrix.row_ptrs[row + 1] - matrix.row_ptrs[row];
                (row, row_nnz)
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|(row_a, nnz_a), (row_b, nnz_b)| {
            nnz_b.cmp(nnz_a).then_with(|| row_a.cmp(row_b))
        });

        let mut row_order = Vec::with_capacity(rows.len());
        let mut bucket_offsets = Vec::new();
        let mut last_nnz = None;
        for (idx, (row, nnz)) in rows.into_iter().enumerate() {
            if last_nnz != Some(nnz) {
                bucket_offsets.push(idx);
                last_nnz = Some(nnz);
            }
            row_order.push(row);
        }
        bucket_offsets.push(row_order.len());

        Self {
            matrix: matrix.clone(),
            row_order,
            bucket_offsets,
        }
    }

    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.matrix.num_cols, "Input vector size mismatch");
        let mut y = vec![T::zero(); self.matrix.num_rows];
        self.matvec_into_buf(x, &mut y);
        y
    }

    pub fn matvec_into_buf(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.matrix.num_cols, "Input vector size mismatch");
        assert_eq!(y.len(), self.matrix.num_rows, "Output vector size mismatch");

        if self.matrix.nnz() < par_nnz_threshold() {
            for &row in &self.row_order {
                y[row] = self.row_dot(row, x);
            }
            return;
        }

        let partials = self
            .row_order
            .par_iter()
            .map(|&row| (row, self.row_dot(row, x)))
            .collect::<Vec<_>>();
        for (row, value) in partials {
            y[row] = value;
        }
    }

    /// Checked row-bucketed SpMV. The row-order metadata is public for
    /// compatibility, so it must be revalidated before indexed execution.
    pub fn try_matvec(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_metadata("row-bucketed CSR SpMV")?;
        validate_checked_vector("row-bucketed CSR SpMV", x, self.matrix.num_cols)?;
        let y = self.matvec(x);
        validate_checked_finite("row-bucketed CSR SpMV", &y).map(|()| y)
    }

    fn validate_checked_metadata(&self, operation: &'static str) -> Result<(), SolverError> {
        self.matrix.validate_structure()?;
        if self.row_order.len() != self.matrix.num_rows {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "row-order length {} != row count {}",
                    self.row_order.len(),
                    self.matrix.num_rows
                ),
            });
        }
        let mut seen = vec![false; self.matrix.num_rows];
        for &row in &self.row_order {
            if row >= self.matrix.num_rows || std::mem::replace(&mut seen[row], true) {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: "row-order must be a permutation of matrix rows".into(),
                });
            }
        }
        if self.bucket_offsets.first().copied() != Some(0)
            || self.bucket_offsets.last().copied() != Some(self.row_order.len())
            || self
                .bucket_offsets
                .windows(2)
                .any(|window| window[0] > window[1] || window[1] > self.row_order.len())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "bucket offsets are not monotone or do not cover rows".into(),
            });
        }
        Ok(())
    }

    pub fn matvec_transpose(&self, x: &[T]) -> Vec<T> {
        self.matrix.matvec_transpose(x)
    }

    pub fn matvec_transpose_into_buf(&self, x: &[T], y: &mut [T]) {
        self.matrix.matvec_transpose_into_buf(x, y);
    }

    pub fn matvec_hermitian(&self, x: &[T]) -> Vec<T> {
        self.matrix.matvec_hermitian(x)
    }

    fn row_dot(&self, row: usize, x: &[T]) -> T {
        let start = self.matrix.row_ptrs[row];
        let end = self.matrix.row_ptrs[row + 1];
        let mut sum = T::zero();
        for idx in start..end {
            let col = self.matrix.col_indices[idx];
            sum += self.matrix.values[idx] * x[col];
        }
        sum
    }
}

impl<T: ComplexField> LinearOperator<T> for RowBucketedCsrMatrix<T> {
    fn num_rows(&self) -> usize {
        self.matrix.num_rows
    }

    fn num_cols(&self) -> usize {
        self.matrix.num_cols
    }

    fn apply(&self, x: &[T]) -> Vec<T> {
        self.matvec(x)
    }

    fn try_apply(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.try_matvec(x)
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_into_buf(x, y);
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_transpose_into_buf(x, y);
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_metadata("row-bucketed CSR transpose SpMV")?;
        self.matrix.try_matvec_transpose(x)
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        self.matvec_hermitian(x)
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_metadata("row-bucketed CSR Hermitian SpMV")?;
        self.matrix.try_matvec_hermitian(x)
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }
}

#[derive(Debug, Clone)]
pub struct SellCSigmaMatrix<T: ComplexField> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub chunk_size: usize,
    pub sigma: usize,
    pub row_order: Vec<usize>,
    pub chunk_offsets: Vec<usize>,
    pub chunk_heights: Vec<usize>,
    pub chunk_widths: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
    /// Plain CSR copy used for transpose/Hermitian products.
    ///
    /// A SELL-native transpose would have to scatter into `y` across the
    /// column-major chunk layout (one `y[col] +=` per padded entry), which is
    /// strictly worse than the row-parallel CSR transpose kernel. The
    /// fallback is therefore deliberate: `apply_transpose_into` and
    /// `apply_hermitian` forward to this CSR clone.
    transpose_fallback: CsrMatrix<T>,
}

impl<T: ComplexField> SellCSigmaMatrix<T> {
    pub fn from_csr(matrix: &CsrMatrix<T>, chunk_size: usize, sigma: usize) -> Self {
        Self::try_from_csr(matrix, chunk_size, sigma)
            .unwrap_or_else(|error| panic!("invalid CSR matrix for SELL-C-sigma: {error}"))
    }

    pub fn try_from_csr(
        matrix: &CsrMatrix<T>,
        chunk_size: usize,
        sigma: usize,
    ) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        Ok(Self::from_csr_unchecked(matrix, chunk_size, sigma))
    }

    fn from_csr_unchecked(matrix: &CsrMatrix<T>, chunk_size: usize, sigma: usize) -> Self {
        let chunk_size = chunk_size.max(1);
        let sigma = sigma.max(chunk_size);
        let row_order = sell_c_sigma_row_order(matrix, chunk_size, sigma);
        let mut chunk_offsets = Vec::new();
        let mut chunk_heights = Vec::new();
        let mut chunk_widths = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for chunk_start in (0..row_order.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(row_order.len());
            let rows = &row_order[chunk_start..chunk_end];
            let height = rows.len();
            let width = rows
                .iter()
                .map(|&row| matrix.row_ptrs[row + 1] - matrix.row_ptrs[row])
                .max()
                .unwrap_or(0);

            chunk_offsets.push(values.len());
            chunk_heights.push(height);
            chunk_widths.push(width);

            for local_col in 0..width {
                for &row in rows {
                    let row_start = matrix.row_ptrs[row];
                    let row_end = matrix.row_ptrs[row + 1];
                    let row_nnz = row_end - row_start;
                    if local_col < row_nnz {
                        let idx = row_start + local_col;
                        col_indices.push(matrix.col_indices[idx]);
                        values.push(matrix.values[idx]);
                    } else {
                        col_indices.push(0);
                        values.push(T::zero());
                    }
                }
            }
        }
        chunk_offsets.push(values.len());

        Self {
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            chunk_size,
            sigma,
            row_order,
            chunk_offsets,
            chunk_heights,
            chunk_widths,
            col_indices,
            values,
            transpose_fallback: matrix.clone(),
        }
    }

    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        let mut y = vec![T::zero(); self.num_rows];
        self.matvec_into_buf(x, &mut y);
        y
    }

    /// Matrix-vector product into a pre-allocated output slice.
    ///
    /// Vectorizes across the C rows of each chunk: the column-major chunk
    /// layout stores the C row values for a given `local_col` contiguously,
    /// so one SIMD lane per row gives gathers/FMAs without any horizontal
    /// reduction inside the chunk. Chunks are processed serially, or in
    /// parallel via rayon once the padded nonzero count and chunk count are
    /// large enough (each chunk owns a disjoint set of rows).
    pub fn matvec_into_buf(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.num_cols, "Input vector size mismatch");
        assert_eq!(y.len(), self.num_rows, "Output vector size mismatch");

        let padded_nnz = self.values.len();
        if padded_nnz >= par_nnz_threshold()
            && self.chunk_heights.len() >= SELL_C_SIGMA_PAR_MIN_CHUNKS
            && sonium_solvers_sys::runtime::bandwidth_threads_for_nnz(padded_nnz) > 1
        {
            self.matvec_parallel_into(x, y);
            return;
        }

        self.matvec_sequential_into(x, y);
    }

    /// Checked SELL-C-sigma SpMV. Public layout metadata is validated before
    /// the SIMD kernels read offsets, padded entries, or row permutations.
    pub fn try_matvec(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_metadata("SELL-C-sigma SpMV")?;
        validate_checked_vector("SELL-C-sigma SpMV", x, self.num_cols)?;
        let y = self.matvec(x);
        validate_checked_finite("SELL-C-sigma SpMV", &y).map(|()| y)
    }

    fn validate_checked_metadata(&self, operation: &'static str) -> Result<(), SolverError> {
        if self.chunk_size == 0 || self.sigma < self.chunk_size {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "invalid SELL chunk-size or sigma metadata".into(),
            });
        }
        if self.row_order.len() != self.num_rows {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!(
                    "row-order length {} != row count {}",
                    self.row_order.len(),
                    self.num_rows
                ),
            });
        }
        let mut seen = vec![false; self.num_rows];
        for &row in &self.row_order {
            if row >= self.num_rows || std::mem::replace(&mut seen[row], true) {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: "row-order must be a permutation of matrix rows".into(),
                });
            }
        }
        if self.col_indices.len() != self.values.len()
            || self
                .col_indices
                .iter()
                .any(|&column| column >= self.num_cols)
            || self.values.iter().any(|value| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "invalid SELL column or value storage".into(),
            });
        }
        if self.chunk_offsets.len() != self.chunk_heights.len().saturating_add(1)
            || self.chunk_heights.len() != self.chunk_widths.len()
            || self.chunk_offsets.first().copied() != Some(0)
            || self.chunk_offsets.last().copied() != Some(self.values.len())
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "invalid SELL chunk metadata lengths or boundaries".into(),
            });
        }
        let mut row_base = 0usize;
        for (chunk, (&height, &width)) in self
            .chunk_heights
            .iter()
            .zip(self.chunk_widths.iter())
            .enumerate()
        {
            if height == 0 || height > self.chunk_size || row_base + height > self.row_order.len() {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("invalid SELL height at chunk {chunk}"),
                });
            }
            let start = self.chunk_offsets[chunk];
            let next = self.chunk_offsets[chunk + 1];
            let padded = height
                .checked_mul(width)
                .ok_or_else(|| SolverError::OperationFailed {
                    operation,
                    reason: format!("SELL chunk {chunk} size overflow"),
                })?;
            if next < start || next - start != padded {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("invalid SELL offset span at chunk {chunk}"),
                });
            }
            row_base += height;
        }
        if row_base != self.row_order.len()
            || self.transpose_fallback.validate_structure().is_err()
            || self.transpose_fallback.num_rows != self.num_rows
            || self.transpose_fallback.num_cols != self.num_cols
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: "SELL row coverage or transpose fallback is invalid".into(),
            });
        }
        Ok(())
    }

    /// Sequential matvec: per-chunk SIMD kernels for the four concrete
    /// scalar types, generic scalar loop otherwise. Each row is written
    /// exactly once (`row_order` is a permutation), so `y` needs no zeroing.
    #[inline]
    fn matvec_sequential_into(&self, x: &[T], y: &mut [T]) {
        if TypeId::of::<T>() == TypeId::of::<f64>() && self.num_cols <= i32::MAX as usize {
            // SAFETY: We just verified T == f64; the casts are
            // layout-identical reinterprets.
            unsafe {
                let x_f64 = &*(x as *const [T] as *const [f64]);
                let y_f64 = &mut *(y as *mut [T] as *mut [f64]);
                let values_f64 = &*(self.values.as_slice() as *const [T] as *const [f64]);
                let mut out = vec![0.0_f64; self.chunk_size];
                for (chunk, (&height, &width)) in self
                    .chunk_heights
                    .iter()
                    .zip(self.chunk_widths.iter())
                    .enumerate()
                {
                    sell_chunk_f64(
                        values_f64,
                        &self.col_indices,
                        self.chunk_offsets[chunk],
                        height,
                        width,
                        x_f64,
                        &mut out,
                    );
                    let row_base = chunk * self.chunk_size;
                    for (local_row, &value) in out.iter().enumerate().take(height) {
                        y_f64[self.row_order[row_base + local_row]] = value;
                    }
                }
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() && self.num_cols <= i32::MAX as usize {
            // SAFETY: We just verified T == f32.
            unsafe {
                let x_f32 = &*(x as *const [T] as *const [f32]);
                let y_f32 = &mut *(y as *mut [T] as *mut [f32]);
                let values_f32 = &*(self.values.as_slice() as *const [T] as *const [f32]);
                let mut out = vec![0.0_f32; self.chunk_size];
                for (chunk, (&height, &width)) in self
                    .chunk_heights
                    .iter()
                    .zip(self.chunk_widths.iter())
                    .enumerate()
                {
                    sell_chunk_f32(
                        values_f32,
                        &self.col_indices,
                        self.chunk_offsets[chunk],
                        height,
                        width,
                        x_f32,
                        &mut out,
                    );
                    let row_base = chunk * self.chunk_size;
                    for (local_row, &value) in out.iter().enumerate().take(height) {
                        y_f32[self.row_order[row_base + local_row]] = value;
                    }
                }
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            // SAFETY: We just verified T == Complex64, whose layout is
            // `(f64, f64)`.
            unsafe {
                let x_c64 = &*(x as *const [T] as *const [Complex64]);
                let y_c64 = &mut *(y as *mut [T] as *mut [Complex64]);
                let values_c64 = &*(self.values.as_slice() as *const [T] as *const [Complex64]);
                let mut out = vec![Complex64::new(0.0, 0.0); self.chunk_size];
                for (chunk, (&height, &width)) in self
                    .chunk_heights
                    .iter()
                    .zip(self.chunk_widths.iter())
                    .enumerate()
                {
                    sell_chunk_complex64(
                        values_c64,
                        &self.col_indices,
                        self.chunk_offsets[chunk],
                        height,
                        width,
                        x_c64,
                        &mut out,
                    );
                    let row_base = chunk * self.chunk_size;
                    for (local_row, &value) in out.iter().enumerate().take(height) {
                        y_c64[self.row_order[row_base + local_row]] = value;
                    }
                }
            }
            return;
        }

        if TypeId::of::<T>() == TypeId::of::<Complex32>() {
            // SAFETY: We just verified T == Complex32, whose layout is
            // `(f32, f32)`.
            unsafe {
                let x_c32 = &*(x as *const [T] as *const [Complex32]);
                let y_c32 = &mut *(y as *mut [T] as *mut [Complex32]);
                let values_c32 = &*(self.values.as_slice() as *const [T] as *const [Complex32]);
                let mut out = vec![Complex32::new(0.0, 0.0); self.chunk_size];
                for (chunk, (&height, &width)) in self
                    .chunk_heights
                    .iter()
                    .zip(self.chunk_widths.iter())
                    .enumerate()
                {
                    sell_chunk_complex32(
                        values_c32,
                        &self.col_indices,
                        self.chunk_offsets[chunk],
                        height,
                        width,
                        x_c32,
                        &mut out,
                    );
                    let row_base = chunk * self.chunk_size;
                    for (local_row, &value) in out.iter().enumerate().take(height) {
                        y_c32[self.row_order[row_base + local_row]] = value;
                    }
                }
            }
            return;
        }

        // Generic scalar fallback for all other types.
        let mut out = vec![T::zero(); self.chunk_size];
        for (chunk, (&height, &width)) in self
            .chunk_heights
            .iter()
            .zip(self.chunk_widths.iter())
            .enumerate()
        {
            sell_chunk_generic(
                &self.values,
                &self.col_indices,
                self.chunk_offsets[chunk],
                height,
                width,
                x,
                &mut out,
            );
            let row_base = chunk * self.chunk_size;
            for (local_row, &value) in out.iter().enumerate().take(height) {
                y[self.row_order[row_base + local_row]] = value;
            }
        }
    }

    /// Parallel matvec: one rayon task per chunk. Each task computes into a
    /// `par_chunks_mut(chunk_size)` slice of a permuted-order buffer (the
    /// chunk boundaries coincide with the C-row buffer splits, and each
    /// chunk owns a disjoint row set), so there are no per-task allocations.
    /// A single sequential scatter converts permuted order to natural row
    /// order afterwards.
    fn matvec_parallel_into(&self, x: &[T], y: &mut [T]) {
        let padded_nnz = self.values.len();

        sonium_solvers_sys::runtime::with_bandwidth_pool_for_nnz(padded_nnz, || {
            if TypeId::of::<T>() == TypeId::of::<f64>() && self.num_cols <= i32::MAX as usize {
                let values_f64 =
                    unsafe { &*(self.values.as_slice() as *const [T] as *const [f64]) };
                let x_f64 = unsafe { &*(x as *const [T] as *const [f64]) };
                let mut permuted = vec![0.0_f64; self.num_rows];
                permuted
                    .par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk, out)| {
                        // SAFETY: SELL invariants guarantee valid indices;
                        // out.len() == the chunk's height by construction.
                        unsafe {
                            sell_chunk_f64(
                                values_f64,
                                &self.col_indices,
                                self.chunk_offsets[chunk],
                                out.len(),
                                self.chunk_widths[chunk],
                                x_f64,
                                out,
                            );
                        }
                    });
                let y_f64 = unsafe { &mut *(y as *mut [T] as *mut [f64]) };
                for (position, &row) in self.row_order.iter().enumerate() {
                    y_f64[row] = permuted[position];
                }
                return;
            }

            if TypeId::of::<T>() == TypeId::of::<f32>() && self.num_cols <= i32::MAX as usize {
                let values_f32 =
                    unsafe { &*(self.values.as_slice() as *const [T] as *const [f32]) };
                let x_f32 = unsafe { &*(x as *const [T] as *const [f32]) };
                let mut permuted = vec![0.0_f32; self.num_rows];
                permuted
                    .par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk, out)| unsafe {
                        sell_chunk_f32(
                            values_f32,
                            &self.col_indices,
                            self.chunk_offsets[chunk],
                            out.len(),
                            self.chunk_widths[chunk],
                            x_f32,
                            out,
                        );
                    });
                let y_f32 = unsafe { &mut *(y as *mut [T] as *mut [f32]) };
                for (position, &row) in self.row_order.iter().enumerate() {
                    y_f32[row] = permuted[position];
                }
                return;
            }

            if TypeId::of::<T>() == TypeId::of::<Complex64>() {
                let values_c64 =
                    unsafe { &*(self.values.as_slice() as *const [T] as *const [Complex64]) };
                let x_c64 = unsafe { &*(x as *const [T] as *const [Complex64]) };
                let mut permuted = vec![Complex64::new(0.0, 0.0); self.num_rows];
                permuted
                    .par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk, out)| unsafe {
                        sell_chunk_complex64(
                            values_c64,
                            &self.col_indices,
                            self.chunk_offsets[chunk],
                            out.len(),
                            self.chunk_widths[chunk],
                            x_c64,
                            out,
                        );
                    });
                let y_c64 = unsafe { &mut *(y as *mut [T] as *mut [Complex64]) };
                for (position, &row) in self.row_order.iter().enumerate() {
                    y_c64[row] = permuted[position];
                }
                return;
            }

            if TypeId::of::<T>() == TypeId::of::<Complex32>() {
                let values_c32 =
                    unsafe { &*(self.values.as_slice() as *const [T] as *const [Complex32]) };
                let x_c32 = unsafe { &*(x as *const [T] as *const [Complex32]) };
                let mut permuted = vec![Complex32::new(0.0, 0.0); self.num_rows];
                permuted
                    .par_chunks_mut(self.chunk_size)
                    .enumerate()
                    .for_each(|(chunk, out)| unsafe {
                        sell_chunk_complex32(
                            values_c32,
                            &self.col_indices,
                            self.chunk_offsets[chunk],
                            out.len(),
                            self.chunk_widths[chunk],
                            x_c32,
                            out,
                        );
                    });
                let y_c32 = unsafe { &mut *(y as *mut [T] as *mut [Complex32]) };
                for (position, &row) in self.row_order.iter().enumerate() {
                    y_c32[row] = permuted[position];
                }
                return;
            }

            // Generic scalar fallback for all other types.
            let mut permuted = vec![T::zero(); self.num_rows];
            permuted
                .par_chunks_mut(self.chunk_size)
                .enumerate()
                .for_each(|(chunk, out)| {
                    sell_chunk_generic(
                        &self.values,
                        &self.col_indices,
                        self.chunk_offsets[chunk],
                        out.len(),
                        self.chunk_widths[chunk],
                        x,
                        out,
                    );
                });
            for (position, &row) in self.row_order.iter().enumerate() {
                y[row] = permuted[position];
            }
        });
    }
}

impl<T: ComplexField> LinearOperator<T> for SellCSigmaMatrix<T> {
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
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        self.matvec_into_buf(x, y);
    }

    /// Transpose product via the CSR fallback (see the `transpose_fallback`
    /// field docs). A SELL-native transpose is intentionally not implemented:
    /// the column-major chunk layout would require a scatter into `y` per
    /// padded entry, which loses to the row-parallel CSR transpose kernel.
    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        self.transpose_fallback.matvec_transpose_into_buf(x, y);
    }

    fn try_apply_transpose(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_metadata("SELL-C-sigma transpose SpMV")?;
        self.transpose_fallback.try_matvec_transpose(x)
    }

    /// Hermitian product via the CSR fallback (same rationale as
    /// [`apply_transpose_into`](Self::apply_transpose_into)).
    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        self.transpose_fallback.matvec_hermitian(x)
    }

    fn try_apply_hermitian(&self, x: &[T]) -> Result<Vec<T>, SolverError> {
        self.validate_checked_metadata("SELL-C-sigma Hermitian SpMV")?;
        self.transpose_fallback.try_matvec_hermitian(x)
    }

    fn transpose_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers_core::traits::AdjointCapability {
        sonium_solvers_core::traits::AdjointCapability::Exact
    }
}

// =============================================================================
// SIMD SpMV kernels for CompactCsrMatrix (u32 indices)
// =============================================================================
//
// These mirror the usize-indexed kernels in `sparse/csr.rs` but read u32
// column indices, which halves index traffic and lets AVX2 load four (f64)
// or eight (f32) indices with a single instruction before the gather.
//
// - x86-64 AVX2: `_mm256_i32gather_pd` / `_mm256_i32gather_ps` for the
//   gathered x[] loads. The u32 -> i32 reinterpretation is exact because the
//   dispatch sites guard `num_cols <= i32::MAX`.
//
// - aarch64 NEON: same structure as csr.rs — no gather instruction, so x[]
//   values are loaded individually and combined with `vsetq_lane_*`; FMA via
//   `vfmaq_*`.
//
// - Scalar fallback for platforms without SIMD support.

/// Sequential driver for the u32-indexed f64 SpMV.
///
/// # Safety
///
/// Caller must ensure the CSR invariants: `row_ptrs` has at least
/// `num_rows + 1` elements, all column indices are valid into `x`, and all
/// row ranges are valid into `values`/`col_indices`. If
/// `allow_avx2_gather` is true, every column index must also be at most
/// `i32::MAX`.
unsafe fn compact_spmv_f64_simd(
    values: &[f64],
    col_indices: &[u32],
    row_ptrs: &[u32],
    x: &[f64],
    y: &mut [f64],
    num_rows: usize,
    allow_avx2_gather: bool,
) {
    for i in 0..num_rows {
        // SAFETY: bounds guaranteed by the caller's CSR invariants.
        unsafe {
            let start = *row_ptrs.get_unchecked(i) as usize;
            let end = *row_ptrs.get_unchecked(i + 1) as usize;
            *y.get_unchecked_mut(i) =
                compact_spmv_row_f64(values, col_indices, x, start, end, allow_avx2_gather);
        }
    }
}

/// Per-row u32-indexed f64 sparse dot-product, dispatching to the best
/// available SIMD path at runtime.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`,
/// and `start..end` must be within bounds for `values` and `col_indices`. If
/// `allow_avx2_gather` is true, those column indices must also be at most
/// `i32::MAX`.
#[inline]
unsafe fn compact_spmv_row_f64(
    values: &[f64],
    col_indices: &[u32],
    x: &[f64],
    start: usize,
    end: usize,
    allow_avx2_gather: bool,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if allow_avx2_gather {
            match selected_spmv_kernel::<f64>() {
                SpmvKernel::Avx2Fma => {
                    // SAFETY: the selected kernel checked runtime AVX2+FMA.
                    return unsafe {
                        compact_spmv_row_f64_avx2_fma(values, col_indices, x, start, end)
                    };
                }
                SpmvKernel::Avx2 => {
                    // SAFETY: the selected kernel checked runtime AVX2.
                    return unsafe {
                        compact_spmv_row_f64_avx2(values, col_indices, x, start, end)
                    };
                }
                _ => {}
            }
        }
        // SAFETY: the caller's CSR bounds invariants are unchanged.
        unsafe { compact_spmv_row_f64_scalar(values, col_indices, x, start, end) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { compact_spmv_row_f64_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { compact_spmv_row_f64_scalar(values, col_indices, x, start, end) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compact_spmv_row_f64_avx2(
    values: &[f64],
    col_indices: &[u32],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::x86_64::*;

    // SAFETY: all AVX2 intrinsics require the avx2 target feature, which is
    // established by the runtime dispatch immediately before this function is
    // called. Slice accesses are within bounds by the CSR invariants
    // documented at the call-site.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();

        let mut acc = _mm256_setzero_pd(); // 4 x f64 accumulator
        let simd_end = start + (len / 4) * 4;

        let mut idx = start;
        while idx < simd_end {
            let v = _mm256_loadu_pd(values.as_ptr().add(idx));

            // Load all four u32 indices with one instruction. The dispatch
            // guard (num_cols <= i32::MAX) makes the u32 -> i32 bit
            // reinterpretation exact for the gather.
            let indices = _mm_loadu_si128(col_indices.as_ptr().add(idx) as *const __m128i);

            // Gather 4 f64 values from x[]; scale factor 8 = sizeof(f64).
            let xv = _mm256_i32gather_pd::<8>(x_ptr, indices);

            acc = _mm256_add_pd(acc, _mm256_mul_pd(v, xv));
            idx += 4;
        }

        // Horizontal sum of the 4-lane accumulator.
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let pair = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(pair, pair);
        let mut sum = _mm_cvtsd_f64(_mm_add_sd(pair, hi64));

        // Scalar remainder (0..3 elements).
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn compact_spmv_row_f64_avx2_fma(
    values: &[f64],
    col_indices: &[u32],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and compact CSR bounds.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();
        let mut acc = _mm256_setzero_pd();
        let simd_end = start + (len / 4) * 4;
        let mut idx = start;
        while idx < simd_end {
            let v = _mm256_loadu_pd(values.as_ptr().add(idx));
            let indices = _mm_loadu_si128(col_indices.as_ptr().add(idx) as *const __m128i);
            let xv = _mm256_i32gather_pd::<8>(x_ptr, indices);
            acc = _mm256_fmadd_pd(v, xv, acc);
            idx += 4;
        }
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let pair = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(pair, pair);
        let mut sum = _mm_cvtsd_f64(_mm_add_sd(pair, hi64));
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
        sum
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn compact_spmv_row_f64_neon(
    values: &[f64],
    col_indices: &[u32],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::aarch64::*;

    let len = end - start;

    // SAFETY: NEON is guaranteed by the #[cfg] gate; slice accesses are
    // within bounds by the CSR invariants documented at the call-site.
    unsafe {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);

        // 4-wide unrolled loop: 2 NEON registers x 2 f64 = 4 elements/iter.
        let simd4_end = start + (len / 4) * 4;
        let mut idx = start;
        while idx < simd4_end {
            let v0 = vld1q_f64(values.as_ptr().add(idx));
            let v1 = vld1q_f64(values.as_ptr().add(idx + 2));

            let j0 = *col_indices.get_unchecked(idx) as usize;
            let j1 = *col_indices.get_unchecked(idx + 1) as usize;
            let j2 = *col_indices.get_unchecked(idx + 2) as usize;
            let j3 = *col_indices.get_unchecked(idx + 3) as usize;

            let xv0 = vsetq_lane_f64::<1>(*x.get_unchecked(j1), vdupq_n_f64(*x.get_unchecked(j0)));
            let xv1 = vsetq_lane_f64::<1>(*x.get_unchecked(j3), vdupq_n_f64(*x.get_unchecked(j2)));

            acc0 = vfmaq_f64(acc0, v0, xv0);
            acc1 = vfmaq_f64(acc1, v1, xv1);

            idx += 4;
        }

        // 2-wide remainder (0 or 2 elements).
        let simd2_end = start + (len / 2) * 2;
        if idx < simd2_end {
            let v = vld1q_f64(values.as_ptr().add(idx));
            let j0 = *col_indices.get_unchecked(idx) as usize;
            let j1 = *col_indices.get_unchecked(idx + 1) as usize;
            let xv = vsetq_lane_f64::<1>(*x.get_unchecked(j1), vdupq_n_f64(*x.get_unchecked(j0)));
            acc0 = vfmaq_f64(acc0, v, xv);
            idx += 2;
        }

        let combined = vaddq_f64(acc0, acc1);
        let mut sum = vaddvq_f64(combined);

        // Scalar remainder (0 or 1 element).
        if idx < end {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn compact_spmv_row_f64_scalar(
    values: &[f64],
    col_indices: &[u32],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    let mut sum = 0.0;
    for idx in start..end {
        // SAFETY: bounds guaranteed by CSR construction invariants.
        unsafe {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
    }
    sum
}

// -----------------------------------------------------------------------------
// Compact u32 — Complex64
// -----------------------------------------------------------------------------

/// Sequential driver for the u32-indexed Complex64 SpMV.
///
/// # Safety
///
/// Same invariants as [`compact_spmv_f64_simd`].
unsafe fn compact_spmv_complex64_simd(
    values: &[Complex64],
    col_indices: &[u32],
    row_ptrs: &[u32],
    x: &[Complex64],
    y: &mut [Complex64],
    num_rows: usize,
) {
    for i in 0..num_rows {
        unsafe {
            let start = *row_ptrs.get_unchecked(i) as usize;
            let end = *row_ptrs.get_unchecked(i + 1) as usize;
            *y.get_unchecked_mut(i) =
                compact_spmv_row_complex64(values, col_indices, x, start, end);
        }
    }
}

/// Per-row u32-indexed Complex64 sparse dot-product.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`.
#[inline]
unsafe fn compact_spmv_row_complex64(
    values: &[Complex64],
    col_indices: &[u32],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    #[cfg(target_arch = "x86_64")]
    {
        match selected_spmv_kernel::<Complex64>() {
            SpmvKernel::Avx2Fma => {
                // SAFETY: the selected kernel checked runtime AVX2+FMA.
                return unsafe {
                    compact_spmv_row_complex64_avx2_fma(values, col_indices, x, start, end)
                };
            }
            SpmvKernel::Avx2 => {
                // SAFETY: the selected kernel checked runtime AVX2.
                return unsafe {
                    compact_spmv_row_complex64_avx2(values, col_indices, x, start, end)
                };
            }
            _ => {}
        }
        // SAFETY: the caller's CSR bounds invariants are unchanged.
        unsafe { compact_spmv_row_complex64_scalar(values, col_indices, x, start, end) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { compact_spmv_row_complex64_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { compact_spmv_row_complex64_scalar(values, col_indices, x, start, end) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compact_spmv_row_complex64_avx2(
    values: &[Complex64],
    col_indices: &[u32],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    use std::arch::x86_64::*;

    // 128-bit SSE2 (always available on x86-64) processes one complex
    // element per iteration with the broadcast-rotate pattern, matching the
    // usize-indexed kernel in csr.rs.
    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;

        let mut acc = _mm_setzero_pd(); // [sum_re, sum_im]
        let neg_mask = _mm_set_pd(0.0, -0.0_f64);

        for idx in start..end {
            let val = _mm_loadu_pd(values_ptr.add(idx * 2));

            let j = *col_indices.get_unchecked(idx) as usize;
            let xv = _mm_loadu_pd(x_ptr.add(j * 2));

            // acc += [re_v, re_v] * [re_x, im_x]
            let re_v = _mm_shuffle_pd::<0x00>(val, val);
            acc = _mm_add_pd(acc, _mm_mul_pd(re_v, xv));

            // acc += [im_v, im_v] * [-im_x, re_x]
            let im_v = _mm_shuffle_pd::<0x03>(val, val);
            let xv_swap = _mm_shuffle_pd::<0x01>(xv, xv);
            let xv_neg_rot = _mm_xor_pd(xv_swap, neg_mask);
            acc = _mm_add_pd(acc, _mm_mul_pd(im_v, xv_neg_rot));
        }

        let mut result = [0.0_f64; 2];
        _mm_storeu_pd(result.as_mut_ptr(), acc);
        Complex64::new(result[0], result[1])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn compact_spmv_row_complex64_avx2_fma(
    values: &[Complex64],
    col_indices: &[u32],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and compact CSR bounds.
    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;
        let mut acc = _mm_setzero_pd();
        let neg_mask = _mm_set_pd(0.0, -0.0_f64);
        for idx in start..end {
            let val = _mm_loadu_pd(values_ptr.add(idx * 2));
            let j = *col_indices.get_unchecked(idx) as usize;
            let xv = _mm_loadu_pd(x_ptr.add(j * 2));
            let re_v = _mm_shuffle_pd::<0x00>(val, val);
            acc = _mm_fmadd_pd(re_v, xv, acc);
            let im_v = _mm_shuffle_pd::<0x03>(val, val);
            let xv_swap = _mm_shuffle_pd::<0x01>(xv, xv);
            let xv_neg_rot = _mm_xor_pd(xv_swap, neg_mask);
            acc = _mm_fmadd_pd(im_v, xv_neg_rot, acc);
        }
        let mut result = [0.0_f64; 2];
        _mm_storeu_pd(result.as_mut_ptr(), acc);
        Complex64::new(result[0], result[1])
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn compact_spmv_row_complex64_neon(
    values: &[Complex64],
    col_indices: &[u32],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    use std::arch::aarch64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;

        let mut acc = vdupq_n_f64(0.0); // [sum_re, sum_im]
        let neg_mask = vsetq_lane_f64::<0>(-1.0, vdupq_n_f64(1.0));

        for idx in start..end {
            let val = vld1q_f64(values_ptr.add(idx * 2));

            let j = *col_indices.get_unchecked(idx) as usize;
            let xv = vld1q_f64(x_ptr.add(j * 2));

            // acc += [re_v, re_v] * [re_x, im_x]
            let re_v = vdupq_laneq_f64::<0>(val);
            acc = vfmaq_f64(acc, re_v, xv);

            // acc += [im_v, im_v] * [-im_x, re_x]
            let im_v = vdupq_laneq_f64::<1>(val);
            let xv_swap = vextq_f64(xv, xv, 1);
            let xv_neg_rot = vmulq_f64(xv_swap, neg_mask);
            acc = vfmaq_f64(acc, im_v, xv_neg_rot);
        }

        let re = vgetq_lane_f64::<0>(acc);
        let im = vgetq_lane_f64::<1>(acc);
        Complex64::new(re, im)
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn compact_spmv_row_complex64_scalar(
    values: &[Complex64],
    col_indices: &[u32],
    x: &[Complex64],
    start: usize,
    end: usize,
) -> Complex64 {
    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for idx in start..end {
        unsafe {
            let j = *col_indices.get_unchecked(idx) as usize;
            let v = *values.get_unchecked(idx);
            let xj = *x.get_unchecked(j);
            sum_re += v.re * xj.re - v.im * xj.im;
            sum_im += v.re * xj.im + v.im * xj.re;
        }
    }
    Complex64::new(sum_re, sum_im)
}

// -----------------------------------------------------------------------------
// Compact u32 — f32
// -----------------------------------------------------------------------------

/// Sequential driver for the u32-indexed f32 SpMV.
///
/// # Safety
///
/// Same invariants as [`compact_spmv_f64_simd`].
/// The AVX2 gather path is enabled only when `allow_avx2_gather` is true;
/// that flag requires all column indices to be at most `i32::MAX`.
unsafe fn compact_spmv_f32_simd(
    values: &[f32],
    col_indices: &[u32],
    row_ptrs: &[u32],
    x: &[f32],
    y: &mut [f32],
    num_rows: usize,
    allow_avx2_gather: bool,
) {
    for i in 0..num_rows {
        unsafe {
            let start = *row_ptrs.get_unchecked(i) as usize;
            let end = *row_ptrs.get_unchecked(i + 1) as usize;
            *y.get_unchecked_mut(i) =
                compact_spmv_row_f32(values, col_indices, x, start, end, allow_avx2_gather);
        }
    }
}

/// Per-row u32-indexed f32 sparse dot-product.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`.
/// If `allow_avx2_gather` is true, those column indices must also be at most
/// `i32::MAX`.
#[inline]
unsafe fn compact_spmv_row_f32(
    values: &[f32],
    col_indices: &[u32],
    x: &[f32],
    start: usize,
    end: usize,
    allow_avx2_gather: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if allow_avx2_gather {
            match selected_spmv_kernel::<f32>() {
                SpmvKernel::Avx2Fma => {
                    // SAFETY: the selected kernel checked runtime AVX2+FMA.
                    return unsafe {
                        compact_spmv_row_f32_avx2_fma(values, col_indices, x, start, end)
                    };
                }
                SpmvKernel::Avx2 => {
                    // SAFETY: the selected kernel checked runtime AVX2.
                    return unsafe {
                        compact_spmv_row_f32_avx2(values, col_indices, x, start, end)
                    };
                }
                _ => {}
            }
        }
        // SAFETY: the caller's CSR bounds invariants are unchanged.
        unsafe { compact_spmv_row_f32_scalar(values, col_indices, x, start, end) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { compact_spmv_row_f32_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { compact_spmv_row_f32_scalar(values, col_indices, x, start, end) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compact_spmv_row_f32_avx2(
    values: &[f32],
    col_indices: &[u32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();

        let mut acc = _mm256_setzero_ps(); // 8 x f32 accumulator
        let simd_end = start + (len / 8) * 8;

        let mut idx = start;
        while idx < simd_end {
            let v = _mm256_loadu_ps(values.as_ptr().add(idx));

            // Load all eight u32 indices with one instruction (see the f64
            // kernel for why the u32 -> i32 reinterpretation is exact).
            let indices = _mm256_loadu_si256(col_indices.as_ptr().add(idx) as *const __m256i);

            // Gather 8 f32 values from x[]; scale factor 4 = sizeof(f32).
            let xv = _mm256_i32gather_ps::<4>(x_ptr, indices);

            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, xv));
            idx += 8;
        }

        // Horizontal sum of the 8-lane accumulator.
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let hi = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, hi);
        let mut sum = _mm_cvtss_f32(final_sum);

        // Scalar remainder (0..7 elements).
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn compact_spmv_row_f32_avx2_fma(
    values: &[f32],
    col_indices: &[u32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and compact CSR bounds.
    unsafe {
        let len = end - start;
        let x_ptr = x.as_ptr();
        let mut acc = _mm256_setzero_ps();
        let simd_end = start + (len / 8) * 8;
        let mut idx = start;
        while idx < simd_end {
            let v = _mm256_loadu_ps(values.as_ptr().add(idx));
            let indices = _mm256_loadu_si256(col_indices.as_ptr().add(idx) as *const __m256i);
            let xv = _mm256_i32gather_ps::<4>(x_ptr, indices);
            acc = _mm256_fmadd_ps(v, xv, acc);
            idx += 8;
        }
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let hi = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, hi);
        let mut sum = _mm_cvtss_f32(final_sum);
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
        sum
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn compact_spmv_row_f32_neon(
    values: &[f32],
    col_indices: &[u32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let len = end - start;

    unsafe {
        let mut acc = vdupq_n_f32(0.0); // 4 x f32 accumulator

        let simd_end = start + (len / 4) * 4;
        let mut idx = start;
        while idx < simd_end {
            let v = vld1q_f32(values.as_ptr().add(idx));

            // NEON has no f32 gather — load individually and combine.
            let j0 = *col_indices.get_unchecked(idx) as usize;
            let j1 = *col_indices.get_unchecked(idx + 1) as usize;
            let j2 = *col_indices.get_unchecked(idx + 2) as usize;
            let j3 = *col_indices.get_unchecked(idx + 3) as usize;

            let mut xv = vdupq_n_f32(*x.get_unchecked(j0));
            xv = vsetq_lane_f32::<1>(*x.get_unchecked(j1), xv);
            xv = vsetq_lane_f32::<2>(*x.get_unchecked(j2), xv);
            xv = vsetq_lane_f32::<3>(*x.get_unchecked(j3), xv);

            acc = vfmaq_f32(acc, v, xv);
            idx += 4;
        }

        let mut sum = vaddvq_f32(acc);

        // Scalar remainder (0..3 elements).
        for idx in simd_end..end {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }

        sum
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn compact_spmv_row_f32_scalar(
    values: &[f32],
    col_indices: &[u32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for idx in start..end {
        unsafe {
            let j = *col_indices.get_unchecked(idx) as usize;
            sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
        }
    }
    sum
}

// -----------------------------------------------------------------------------
// Compact u32 — Complex32
// -----------------------------------------------------------------------------

/// Sequential driver for the u32-indexed Complex32 SpMV.
///
/// # Safety
///
/// Same invariants as [`compact_spmv_f64_simd`].
unsafe fn compact_spmv_complex32_simd(
    values: &[Complex32],
    col_indices: &[u32],
    row_ptrs: &[u32],
    x: &[Complex32],
    y: &mut [Complex32],
    num_rows: usize,
) {
    for i in 0..num_rows {
        unsafe {
            let start = *row_ptrs.get_unchecked(i) as usize;
            let end = *row_ptrs.get_unchecked(i + 1) as usize;
            *y.get_unchecked_mut(i) =
                compact_spmv_row_complex32(values, col_indices, x, start, end);
        }
    }
}

/// Per-row u32-indexed Complex32 sparse dot-product.
///
/// # Safety
///
/// All indices in `col_indices[start..end]` must be valid indices into `x`.
#[inline]
unsafe fn compact_spmv_row_complex32(
    values: &[Complex32],
    col_indices: &[u32],
    x: &[Complex32],
    start: usize,
    end: usize,
) -> Complex32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { compact_spmv_row_complex32_neon(values, col_indices, x, start, end) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        unsafe { compact_spmv_row_complex32_scalar(values, col_indices, x, start, end) }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn compact_spmv_row_complex32_neon(
    values: &[Complex32],
    col_indices: &[u32],
    x: &[Complex32],
    start: usize,
    end: usize,
) -> Complex32 {
    use std::arch::aarch64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f32;
        let x_ptr = x.as_ptr() as *const f32;

        let mut acc = vdup_n_f32(0.0); // [sum_re, sum_im]
        let neg_mask = vset_lane_f32::<0>(-1.0, vdup_n_f32(1.0));

        for idx in start..end {
            let val = vld1_f32(values_ptr.add(idx * 2));

            let j = *col_indices.get_unchecked(idx) as usize;
            let xv = vld1_f32(x_ptr.add(j * 2));

            // acc += [re_v, re_v] * [re_x, im_x]
            let re_v = vdup_lane_f32::<0>(val);
            acc = vfma_f32(acc, re_v, xv);

            // acc += [im_v, im_v] * [-im_x, re_x]
            let im_v = vdup_lane_f32::<1>(val);
            let xv_swap = vext_f32::<1>(xv, xv);
            let xv_neg_rot = vmul_f32(xv_swap, neg_mask);
            acc = vfma_f32(acc, im_v, xv_neg_rot);
        }

        let re = vget_lane_f32::<0>(acc);
        let im = vget_lane_f32::<1>(acc);
        Complex32::new(re, im)
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
unsafe fn compact_spmv_row_complex32_scalar(
    values: &[Complex32],
    col_indices: &[u32],
    x: &[Complex32],
    start: usize,
    end: usize,
) -> Complex32 {
    let mut sum_re = 0.0_f32;
    let mut sum_im = 0.0_f32;
    for idx in start..end {
        unsafe {
            let j = *col_indices.get_unchecked(idx) as usize;
            let v = *values.get_unchecked(idx);
            let xj = *x.get_unchecked(j);
            sum_re += v.re * xj.re - v.im * xj.im;
            sum_im += v.re * xj.im + v.im * xj.re;
        }
    }
    Complex32::new(sum_re, sum_im)
}

// =============================================================================
// SIMD SpMV kernels for SellCSigmaMatrix
// =============================================================================
//
// The SELL-C-sigma chunk layout is column-major within each C-row chunk:
// for a fixed `local_col`, the C row values (and their column indices) are
// contiguous. The kernels therefore vectorize ACROSS rows — one SIMD lane
// per row — accumulating a vector of C partial dot products over the chunk
// width and storing it without any horizontal reduction. Rows beyond the
// last full vector block (chunked tail when `height % lanes != 0`) fall back
// to the scalar row loop. Padded entries carry value zero, so they add
// exactly 0.0 to the accumulator (finite x[0] assumed, as elsewhere).

/// Generic scalar chunk kernel: one dot product per row, iterating
/// `local_col` with stride `height` (the column-major chunk layout).
fn sell_chunk_generic<T: ComplexField>(
    values: &[T],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[T],
    out: &mut [T],
) {
    for (local_row, value) in out.iter_mut().enumerate().take(height) {
        let mut sum = T::zero();
        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            sum += values[idx] * x[col_indices[idx]];
        }
        *value = sum;
    }
}

/// Scalar accumulation for rows `from_row..height` of an f64 chunk.
///
/// Shared by the platform kernels (tail rows) and used as the whole-chunk
/// fallback on platforms without SIMD.
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f64`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_rows_f64_scalar(
    values: &[f64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f64],
    out: &mut [f64],
    from_row: usize,
) {
    for (local_row, out_value) in out.iter_mut().enumerate().take(height).skip(from_row) {
        let mut sum = 0.0_f64;
        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            unsafe {
                sum +=
                    *values.get_unchecked(idx) * *x.get_unchecked(*col_indices.get_unchecked(idx));
            }
        }
        *out_value = sum;
    }
}

/// Compute one SELL-C-sigma chunk of the f64 matvec: `out[local_row]` is the
/// dot product of chunk row `local_row` with `x`, for `local_row < height`.
///
/// # Safety
///
/// Caller must ensure `offset + width * height <= values.len()`, that all
/// column indices in the chunk are valid into `x`, and that
/// `out.len() >= height`.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f64(
    values: &[f64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f64],
    out: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        match selected_spmv_kernel::<f64>() {
            SpmvKernel::Avx2Fma => {
                // SAFETY: the selected kernel checked runtime AVX2+FMA.
                return unsafe {
                    sell_chunk_f64_avx2_fma(values, col_indices, offset, height, width, x, out)
                };
            }
            SpmvKernel::Avx2 => {
                // SAFETY: the selected kernel checked runtime AVX2.
                return unsafe {
                    sell_chunk_f64_avx2(values, col_indices, offset, height, width, x, out)
                };
            }
            _ => {}
        }
        // SAFETY: the caller's SELL bounds invariants are unchanged.
        unsafe { sell_rows_f64_scalar(values, col_indices, offset, height, width, x, out, 0) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { sell_chunk_f64_neon(values, col_indices, offset, height, width, x, out) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { sell_rows_f64_scalar(values, col_indices, offset, height, width, x, out, 0) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f64_avx2(
    values: &[f64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f64],
    out: &mut [f64],
) {
    use std::arch::x86_64::*;

    // SAFETY: AVX2 is established by the runtime dispatch immediately before
    // this function is called; slice accesses are within bounds by the SELL
    // invariants documented at the call-site. The usize -> i32 index
    // narrowing is exact because the dispatch checks every chunk index.
    unsafe {
        let x_ptr = x.as_ptr();
        let mut lanes = [0.0_f64; 4];
        let mut row_block = 0usize;

        // 4 rows per block: one 4-lane accumulator, one lane per row.
        while row_block + 4 <= height {
            let mut acc = _mm256_setzero_pd();
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v = _mm256_loadu_pd(values.as_ptr().add(base));

                let i0 = *col_indices.get_unchecked(base) as i32;
                let i1 = *col_indices.get_unchecked(base + 1) as i32;
                let i2 = *col_indices.get_unchecked(base + 2) as i32;
                let i3 = *col_indices.get_unchecked(base + 3) as i32;
                let indices = _mm_set_epi32(i3, i2, i1, i0);

                let xv = _mm256_i32gather_pd::<8>(x_ptr, indices);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(v, xv));
            }
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
            out[row_block..row_block + 4].copy_from_slice(&lanes);
            row_block += 4;
        }

        // Scalar tail rows (height not a multiple of 4).
        sell_rows_f64_scalar(
            values,
            col_indices,
            offset,
            height,
            width,
            x,
            out,
            row_block,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f64_avx2_fma(
    values: &[f64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f64],
    out: &mut [f64],
) {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and SELL bounds.
    unsafe {
        let x_ptr = x.as_ptr();
        let mut lanes = [0.0_f64; 4];
        let mut row_block = 0usize;
        while row_block + 4 <= height {
            let mut acc = _mm256_setzero_pd();
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v = _mm256_loadu_pd(values.as_ptr().add(base));
                let i0 = *col_indices.get_unchecked(base) as i32;
                let i1 = *col_indices.get_unchecked(base + 1) as i32;
                let i2 = *col_indices.get_unchecked(base + 2) as i32;
                let i3 = *col_indices.get_unchecked(base + 3) as i32;
                let indices = _mm_set_epi32(i3, i2, i1, i0);
                let xv = _mm256_i32gather_pd::<8>(x_ptr, indices);
                acc = _mm256_fmadd_pd(v, xv, acc);
            }
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
            out[row_block..row_block + 4].copy_from_slice(&lanes);
            row_block += 4;
        }
        sell_rows_f64_scalar(
            values,
            col_indices,
            offset,
            height,
            width,
            x,
            out,
            row_block,
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f64_neon(
    values: &[f64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f64],
    out: &mut [f64],
) {
    use std::arch::aarch64::*;

    // SAFETY: NEON is guaranteed by the #[cfg] gate; slice accesses are
    // within bounds by the SELL invariants documented at the call-site.
    unsafe {
        let mut row_block = 0usize;

        // 4 rows per block: two 2-lane accumulators, one lane per row.
        while row_block + 4 <= height {
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v0 = vld1q_f64(values.as_ptr().add(base));
                let v1 = vld1q_f64(values.as_ptr().add(base + 2));

                let j0 = *col_indices.get_unchecked(base);
                let j1 = *col_indices.get_unchecked(base + 1);
                let j2 = *col_indices.get_unchecked(base + 2);
                let j3 = *col_indices.get_unchecked(base + 3);

                let xv0 =
                    vsetq_lane_f64::<1>(*x.get_unchecked(j1), vdupq_n_f64(*x.get_unchecked(j0)));
                let xv1 =
                    vsetq_lane_f64::<1>(*x.get_unchecked(j3), vdupq_n_f64(*x.get_unchecked(j2)));

                acc0 = vfmaq_f64(acc0, v0, xv0);
                acc1 = vfmaq_f64(acc1, v1, xv1);
            }
            vst1q_f64(out.as_mut_ptr().add(row_block), acc0);
            vst1q_f64(out.as_mut_ptr().add(row_block + 2), acc1);
            row_block += 4;
        }

        // 2-row block.
        if row_block + 2 <= height {
            let mut acc = vdupq_n_f64(0.0);
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v = vld1q_f64(values.as_ptr().add(base));
                let j0 = *col_indices.get_unchecked(base);
                let j1 = *col_indices.get_unchecked(base + 1);
                let xv =
                    vsetq_lane_f64::<1>(*x.get_unchecked(j1), vdupq_n_f64(*x.get_unchecked(j0)));
                acc = vfmaq_f64(acc, v, xv);
            }
            vst1q_f64(out.as_mut_ptr().add(row_block), acc);
            row_block += 2;
        }

        // Scalar tail row.
        sell_rows_f64_scalar(
            values,
            col_indices,
            offset,
            height,
            width,
            x,
            out,
            row_block,
        );
    }
}

/// Scalar accumulation for rows `from_row..height` of an f32 chunk.
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f32`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_rows_f32_scalar(
    values: &[f32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f32],
    out: &mut [f32],
    from_row: usize,
) {
    for (local_row, out_value) in out.iter_mut().enumerate().take(height).skip(from_row) {
        let mut sum = 0.0_f32;
        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            unsafe {
                sum +=
                    *values.get_unchecked(idx) * *x.get_unchecked(*col_indices.get_unchecked(idx));
            }
        }
        *out_value = sum;
    }
}

/// Compute one SELL-C-sigma chunk of the f32 matvec.
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f64`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f32(
    values: &[f32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        match selected_spmv_kernel::<f32>() {
            SpmvKernel::Avx2Fma => {
                // SAFETY: the selected kernel checked runtime AVX2+FMA.
                return unsafe {
                    sell_chunk_f32_avx2_fma(values, col_indices, offset, height, width, x, out)
                };
            }
            SpmvKernel::Avx2 => {
                // SAFETY: the selected kernel checked runtime AVX2.
                return unsafe {
                    sell_chunk_f32_avx2(values, col_indices, offset, height, width, x, out)
                };
            }
            _ => {}
        }
        // SAFETY: the caller's SELL bounds invariants are unchanged.
        unsafe { sell_rows_f32_scalar(values, col_indices, offset, height, width, x, out, 0) }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { sell_chunk_f32_neon(values, col_indices, offset, height, width, x, out) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe { sell_rows_f32_scalar(values, col_indices, offset, height, width, x, out, 0) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f32_avx2(
    values: &[f32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    // SAFETY: same as the f64 AVX2 chunk kernel.
    unsafe {
        let x_ptr = x.as_ptr();
        let mut lanes = [0.0_f32; 8];
        let mut row_block = 0usize;

        // 8 rows per block: one 8-lane accumulator, one lane per row.
        while row_block + 8 <= height {
            let mut acc = _mm256_setzero_ps();
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v = _mm256_loadu_ps(values.as_ptr().add(base));

                let i0 = *col_indices.get_unchecked(base) as i32;
                let i1 = *col_indices.get_unchecked(base + 1) as i32;
                let i2 = *col_indices.get_unchecked(base + 2) as i32;
                let i3 = *col_indices.get_unchecked(base + 3) as i32;
                let i4 = *col_indices.get_unchecked(base + 4) as i32;
                let i5 = *col_indices.get_unchecked(base + 5) as i32;
                let i6 = *col_indices.get_unchecked(base + 6) as i32;
                let i7 = *col_indices.get_unchecked(base + 7) as i32;
                let indices = _mm256_set_epi32(i7, i6, i5, i4, i3, i2, i1, i0);

                let xv = _mm256_i32gather_ps::<4>(x_ptr, indices);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, xv));
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
            out[row_block..row_block + 8].copy_from_slice(&lanes);
            row_block += 8;
        }

        // Scalar tail rows (height not a multiple of 8).
        sell_rows_f32_scalar(
            values,
            col_indices,
            offset,
            height,
            width,
            x,
            out,
            row_block,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f32_avx2_fma(
    values: &[f32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and SELL bounds.
    unsafe {
        let x_ptr = x.as_ptr();
        let mut lanes = [0.0_f32; 8];
        let mut row_block = 0usize;
        while row_block + 8 <= height {
            let mut acc = _mm256_setzero_ps();
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v = _mm256_loadu_ps(values.as_ptr().add(base));
                let i0 = *col_indices.get_unchecked(base) as i32;
                let i1 = *col_indices.get_unchecked(base + 1) as i32;
                let i2 = *col_indices.get_unchecked(base + 2) as i32;
                let i3 = *col_indices.get_unchecked(base + 3) as i32;
                let i4 = *col_indices.get_unchecked(base + 4) as i32;
                let i5 = *col_indices.get_unchecked(base + 5) as i32;
                let i6 = *col_indices.get_unchecked(base + 6) as i32;
                let i7 = *col_indices.get_unchecked(base + 7) as i32;
                let indices = _mm256_set_epi32(i7, i6, i5, i4, i3, i2, i1, i0);
                let xv = _mm256_i32gather_ps::<4>(x_ptr, indices);
                acc = _mm256_fmadd_ps(v, xv, acc);
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
            out[row_block..row_block + 8].copy_from_slice(&lanes);
            row_block += 8;
        }
        sell_rows_f32_scalar(
            values,
            col_indices,
            offset,
            height,
            width,
            x,
            out,
            row_block,
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_f32_neon(
    values: &[f32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    // SAFETY: NEON is guaranteed by the #[cfg] gate; slice accesses are
    // within bounds by the SELL invariants documented at the call-site.
    unsafe {
        let mut row_block = 0usize;

        // 4 rows per block: one 4-lane accumulator, one lane per row.
        while row_block + 4 <= height {
            let mut acc = vdupq_n_f32(0.0);
            for local_col in 0..width {
                let base = offset + local_col * height + row_block;
                let v = vld1q_f32(values.as_ptr().add(base));

                let j0 = *col_indices.get_unchecked(base);
                let j1 = *col_indices.get_unchecked(base + 1);
                let j2 = *col_indices.get_unchecked(base + 2);
                let j3 = *col_indices.get_unchecked(base + 3);

                let mut xv = vdupq_n_f32(*x.get_unchecked(j0));
                xv = vsetq_lane_f32::<1>(*x.get_unchecked(j1), xv);
                xv = vsetq_lane_f32::<2>(*x.get_unchecked(j2), xv);
                xv = vsetq_lane_f32::<3>(*x.get_unchecked(j3), xv);

                acc = vfmaq_f32(acc, v, xv);
            }
            vst1q_f32(out.as_mut_ptr().add(row_block), acc);
            row_block += 4;
        }

        // Scalar tail rows (height not a multiple of 4).
        sell_rows_f32_scalar(
            values,
            col_indices,
            offset,
            height,
            width,
            x,
            out,
            row_block,
        );
    }
}

/// Compute one SELL-C-sigma chunk of the Complex64 matvec.
///
/// Each row is processed with the broadcast-rotate complex kernel from
/// csr.rs (one complex element per iteration over `local_col`, strided
/// loads through the column-major chunk layout).
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f64`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_complex64(
    values: &[Complex64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex64],
    out: &mut [Complex64],
) {
    for (local_row, value) in out.iter_mut().enumerate().take(height) {
        *value =
            unsafe { sell_row_complex64(values, col_indices, offset, height, width, x, local_row) };
    }
}

/// Per-row Complex64 chunk dot-product over `local_col`.
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f64`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex64(
    values: &[Complex64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex64],
    local_row: usize,
) -> Complex64 {
    #[cfg(target_arch = "x86_64")]
    {
        match selected_spmv_kernel::<Complex64>() {
            SpmvKernel::Avx2Fma => {
                // SAFETY: the selected kernel checked runtime AVX2+FMA.
                return unsafe {
                    sell_row_complex64_avx2_fma(
                        values,
                        col_indices,
                        offset,
                        height,
                        width,
                        x,
                        local_row,
                    )
                };
            }
            SpmvKernel::Avx2 => {
                // SAFETY: the selected kernel checked runtime AVX2.
                return unsafe {
                    sell_row_complex64_avx2(
                        values,
                        col_indices,
                        offset,
                        height,
                        width,
                        x,
                        local_row,
                    )
                };
            }
            _ => {}
        }
        // SAFETY: the caller's SELL bounds invariants are unchanged.
        unsafe {
            sell_row_complex64_scalar(values, col_indices, offset, height, width, x, local_row)
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { sell_row_complex64_neon(values, col_indices, offset, height, width, x, local_row) }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        unsafe {
            sell_row_complex64_scalar(values, col_indices, offset, height, width, x, local_row)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex64_avx2(
    values: &[Complex64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex64],
    local_row: usize,
) -> Complex64 {
    use std::arch::x86_64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;

        let mut acc = _mm_setzero_pd(); // [sum_re, sum_im]
        let neg_mask = _mm_set_pd(0.0, -0.0_f64);

        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            let val = _mm_loadu_pd(values_ptr.add(idx * 2));

            let j = *col_indices.get_unchecked(idx);
            let xv = _mm_loadu_pd(x_ptr.add(j * 2));

            let re_v = _mm_shuffle_pd::<0x00>(val, val);
            acc = _mm_add_pd(acc, _mm_mul_pd(re_v, xv));

            let im_v = _mm_shuffle_pd::<0x03>(val, val);
            let xv_swap = _mm_shuffle_pd::<0x01>(xv, xv);
            let xv_neg_rot = _mm_xor_pd(xv_swap, neg_mask);
            acc = _mm_add_pd(acc, _mm_mul_pd(im_v, xv_neg_rot));
        }

        let mut result = [0.0_f64; 2];
        _mm_storeu_pd(result.as_mut_ptr(), acc);
        Complex64::new(result[0], result[1])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex64_avx2_fma(
    values: &[Complex64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex64],
    local_row: usize,
) -> Complex64 {
    use std::arch::x86_64::*;

    // SAFETY: the caller checked AVX2+FMA and SELL bounds.
    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;
        let mut acc = _mm_setzero_pd();
        let neg_mask = _mm_set_pd(0.0, -0.0_f64);
        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            let val = _mm_loadu_pd(values_ptr.add(idx * 2));
            let j = *col_indices.get_unchecked(idx);
            let xv = _mm_loadu_pd(x_ptr.add(j * 2));
            let re_v = _mm_shuffle_pd::<0x00>(val, val);
            acc = _mm_fmadd_pd(re_v, xv, acc);
            let im_v = _mm_shuffle_pd::<0x03>(val, val);
            let xv_swap = _mm_shuffle_pd::<0x01>(xv, xv);
            let xv_neg_rot = _mm_xor_pd(xv_swap, neg_mask);
            acc = _mm_fmadd_pd(im_v, xv_neg_rot, acc);
        }
        let mut result = [0.0_f64; 2];
        _mm_storeu_pd(result.as_mut_ptr(), acc);
        Complex64::new(result[0], result[1])
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex64_neon(
    values: &[Complex64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex64],
    local_row: usize,
) -> Complex64 {
    use std::arch::aarch64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f64;
        let x_ptr = x.as_ptr() as *const f64;

        let mut acc = vdupq_n_f64(0.0); // [sum_re, sum_im]
        let neg_mask = vsetq_lane_f64::<0>(-1.0, vdupq_n_f64(1.0));

        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            let val = vld1q_f64(values_ptr.add(idx * 2));

            let j = *col_indices.get_unchecked(idx);
            let xv = vld1q_f64(x_ptr.add(j * 2));

            let re_v = vdupq_laneq_f64::<0>(val);
            acc = vfmaq_f64(acc, re_v, xv);

            let im_v = vdupq_laneq_f64::<1>(val);
            let xv_swap = vextq_f64(xv, xv, 1);
            let xv_neg_rot = vmulq_f64(xv_swap, neg_mask);
            acc = vfmaq_f64(acc, im_v, xv_neg_rot);
        }

        let re = vgetq_lane_f64::<0>(acc);
        let im = vgetq_lane_f64::<1>(acc);
        Complex64::new(re, im)
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex64_scalar(
    values: &[Complex64],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex64],
    local_row: usize,
) -> Complex64 {
    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for local_col in 0..width {
        let idx = offset + local_col * height + local_row;
        unsafe {
            let j = *col_indices.get_unchecked(idx);
            let v = *values.get_unchecked(idx);
            let xj = *x.get_unchecked(j);
            sum_re += v.re * xj.re - v.im * xj.im;
            sum_im += v.re * xj.im + v.im * xj.re;
        }
    }
    Complex64::new(sum_re, sum_im)
}

/// Compute one SELL-C-sigma chunk of the Complex32 matvec.
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f64`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_chunk_complex32(
    values: &[Complex32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex32],
    out: &mut [Complex32],
) {
    for (local_row, value) in out.iter_mut().enumerate().take(height) {
        *value =
            unsafe { sell_row_complex32(values, col_indices, offset, height, width, x, local_row) };
    }
}

/// Per-row Complex32 chunk dot-product over `local_col`.
///
/// # Safety
///
/// Same invariants as [`sell_chunk_f64`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex32(
    values: &[Complex32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex32],
    local_row: usize,
) -> Complex32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { sell_row_complex32_neon(values, col_indices, offset, height, width, x, local_row) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        unsafe {
            sell_row_complex32_scalar(values, col_indices, offset, height, width, x, local_row)
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex32_neon(
    values: &[Complex32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex32],
    local_row: usize,
) -> Complex32 {
    use std::arch::aarch64::*;

    unsafe {
        let values_ptr = values.as_ptr() as *const f32;
        let x_ptr = x.as_ptr() as *const f32;

        let mut acc = vdup_n_f32(0.0); // [sum_re, sum_im]
        let neg_mask = vset_lane_f32::<0>(-1.0, vdup_n_f32(1.0));

        for local_col in 0..width {
            let idx = offset + local_col * height + local_row;
            let val = vld1_f32(values_ptr.add(idx * 2));

            let j = *col_indices.get_unchecked(idx);
            let xv = vld1_f32(x_ptr.add(j * 2));

            let re_v = vdup_lane_f32::<0>(val);
            acc = vfma_f32(acc, re_v, xv);

            let im_v = vdup_lane_f32::<1>(val);
            let xv_swap = vext_f32::<1>(xv, xv);
            let xv_neg_rot = vmul_f32(xv_swap, neg_mask);
            acc = vfma_f32(acc, im_v, xv_neg_rot);
        }

        let re = vget_lane_f32::<0>(acc);
        let im = vget_lane_f32::<1>(acc);
        Complex32::new(re, im)
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sell_row_complex32_scalar(
    values: &[Complex32],
    col_indices: &[usize],
    offset: usize,
    height: usize,
    width: usize,
    x: &[Complex32],
    local_row: usize,
) -> Complex32 {
    let mut sum_re = 0.0_f32;
    let mut sum_im = 0.0_f32;
    for local_col in 0..width {
        let idx = offset + local_col * height + local_row;
        unsafe {
            let j = *col_indices.get_unchecked(idx);
            let v = *values.get_unchecked(idx);
            let xj = *x.get_unchecked(j);
            sum_re += v.re * xj.re - v.im * xj.im;
            sum_im += v.re * xj.im + v.im * xj.re;
        }
    }
    Complex32::new(sum_re, sum_im)
}

fn choose_format(
    analysis: &SpmvAnalysis,
    config: &SpmvPlanConfig,
) -> (SpmvPlanFormat, Vec<String>) {
    let mut reasons = Vec::new();
    match config.preference {
        SpmvPlanPreference::Csr => {
            reasons.push("forced CSR by config".to_string());
            return (SpmvPlanFormat::Csr, reasons);
        }
        SpmvPlanPreference::CompactCsrU32 => {
            if analysis.can_use_u32_indices {
                reasons.push("forced compact u32 CSR by config".to_string());
                return (SpmvPlanFormat::CompactCsrU32, reasons);
            }
            reasons.push("compact u32 CSR requested, but indices do not fit u32".to_string());
            return (SpmvPlanFormat::Csr, reasons);
        }
        SpmvPlanPreference::NumaCsr => {
            reasons.push("forced NUMA CSR by config".to_string());
            return (SpmvPlanFormat::NumaCsr, reasons);
        }
        SpmvPlanPreference::RowBucketedCsr => {
            if let Some(row_bucket) = analysis.row_bucket {
                reasons.push(format!(
                    "forced row-bucketed CSR by config: buckets={}, row_nnz_cv={:.3}",
                    row_bucket.bucket_count, analysis.row_nnz_cv
                ));
                return (SpmvPlanFormat::RowBucketedCsr, reasons);
            }
            reasons.push("row-bucketed CSR requested, but matrix has no rows".to_string());
            return (SpmvPlanFormat::Csr, reasons);
        }
        SpmvPlanPreference::SellCSigma => {
            if let Some(sell) = analysis.sell_c_sigma {
                reasons.push(format!(
                    "forced SELL-C-sigma by config: C={}, sigma={}, fill_ratio={:.3}",
                    sell.chunk_size, sell.sigma, sell.fill_ratio
                ));
                return (SpmvPlanFormat::SellCSigma, reasons);
            }
            reasons.push("SELL-C-sigma requested, but matrix has no rows".to_string());
            return (SpmvPlanFormat::Csr, reasons);
        }
        SpmvPlanPreference::Bsr2 => {
            return forced_bsr_format(analysis.bsr2, SpmvPlanFormat::Bsr2, &mut reasons);
        }
        SpmvPlanPreference::Bsr3 => {
            return forced_bsr_format(analysis.bsr3, SpmvPlanFormat::Bsr3, &mut reasons);
        }
        SpmvPlanPreference::Bsr4 => {
            return forced_bsr_format(analysis.bsr4, SpmvPlanFormat::Bsr4, &mut reasons);
        }
        SpmvPlanPreference::Auto => {}
    }

    let numa_partitions = config
        .numa_partitions
        .unwrap_or_else(detect_numa_partitions);
    if numa_partitions > 1 && analysis.nnz >= config.min_nnz_for_numa {
        reasons.push(format!(
            "auto selected NUMA CSR: partitions={numa_partitions}, nnz={} >= threshold={}",
            analysis.nnz, config.min_nnz_for_numa
        ));
        return (SpmvPlanFormat::NumaCsr, reasons);
    }

    if let Some((format, bsr)) = best_bsr_candidate(analysis) {
        let saved = bsr.saved_bytes_per_apply as usize;
        if config.expected_reuse >= config.min_reuse_for_bsr
            && saved >= config.min_saved_bytes_for_bsr
            && bsr.block_fill_ratio >= config.min_bsr_block_fill_ratio
        {
            reasons.push(format!(
                "auto selected BSR{}: expected_reuse={} >= {}, saved_bytes/apply={} >= {}, block_fill_ratio={:.3} >= {:.3}",
                bsr.block_size,
                config.expected_reuse,
                config.min_reuse_for_bsr,
                saved,
                config.min_saved_bytes_for_bsr,
                bsr.block_fill_ratio,
                config.min_bsr_block_fill_ratio
            ));
            return (format, reasons);
        }
        reasons.push(format!(
            "kept CSR over BSR{}: expected_reuse={}, saved_bytes/apply={}, block_fill_ratio={:.3} below BSR thresholds",
            bsr.block_size, config.expected_reuse, saved, bsr.block_fill_ratio
        ));
    } else {
        reasons.push("kept CSR: no 2/3/4 block structure saves bytes".to_string());
    }

    if let Some(sell) = analysis.sell_c_sigma {
        if config.expected_reuse >= config.min_reuse_for_sell_c_sigma
            && analysis.nnz >= config.min_nnz_for_sell_c_sigma
            && sell.fill_ratio >= config.min_sell_c_sigma_fill_ratio
            && sell.padding_overhead_nnz <= analysis.nnz
        {
            reasons.push(format!(
                "auto selected SELL-C-sigma: expected_reuse={} >= {}, nnz={} >= {}, fill_ratio={:.3} >= {:.3}, padding_overhead_nnz={}",
                config.expected_reuse,
                config.min_reuse_for_sell_c_sigma,
                analysis.nnz,
                config.min_nnz_for_sell_c_sigma,
                sell.fill_ratio,
                config.min_sell_c_sigma_fill_ratio,
                sell.padding_overhead_nnz
            ));
            return (SpmvPlanFormat::SellCSigma, reasons);
        }
        reasons.push(format!(
            "kept CSR over SELL-C-sigma: expected_reuse={}, nnz={}, fill_ratio={:.3}, padding_overhead_nnz={} below SELL thresholds",
            config.expected_reuse, analysis.nnz, sell.fill_ratio, sell.padding_overhead_nnz
        ));
    }

    if let Some(row_bucket) = analysis.row_bucket {
        if config.expected_reuse >= config.min_reuse_for_row_bucket
            && analysis.nnz >= config.min_nnz_for_row_bucket
            && analysis.row_nnz_cv >= config.min_row_nnz_cv_for_row_bucket
        {
            reasons.push(format!(
                "auto selected row-bucketed CSR: expected_reuse={} >= {}, nnz={} >= {}, row_nnz_cv={:.3} >= {:.3}",
                config.expected_reuse,
                config.min_reuse_for_row_bucket,
                analysis.nnz,
                config.min_nnz_for_row_bucket,
                analysis.row_nnz_cv,
                config.min_row_nnz_cv_for_row_bucket
            ));
            return (SpmvPlanFormat::RowBucketedCsr, reasons);
        }
        reasons.push(format!(
            "kept CSR over row-bucketed CSR: expected_reuse={}, nnz={}, row_nnz_cv={:.3}, buckets={} below row-bucket thresholds",
            config.expected_reuse, analysis.nnz, analysis.row_nnz_cv, row_bucket.bucket_count
        ));
    }

    if analysis.can_use_u32_indices {
        let saved = analysis.saved_index_bytes_per_apply.unwrap_or(0) as usize;
        if config.expected_reuse >= config.min_reuse_for_compact_indices
            && saved >= config.min_saved_index_bytes_for_compact
        {
            reasons.push(format!(
                "auto selected compact u32 CSR: expected_reuse={} >= {}, saved_index_bytes/apply={} >= {}",
                config.expected_reuse,
                config.min_reuse_for_compact_indices,
                saved,
                config.min_saved_index_bytes_for_compact
            ));
            return (SpmvPlanFormat::CompactCsrU32, reasons);
        }
        reasons.push(format!(
            "kept CSR: u32 indices fit, but expected_reuse={} or saved_index_bytes/apply={} below compact thresholds",
            config.expected_reuse, saved
        ));
    } else {
        reasons.push("kept CSR: matrix indices do not fit u32".to_string());
    }

    if numa_partitions <= 1 {
        reasons.push("kept CSR: only one NUMA partition detected/requested".to_string());
    } else if analysis.nnz < config.min_nnz_for_numa {
        reasons.push(format!(
            "kept CSR: nnz={} below NUMA threshold={}",
            analysis.nnz, config.min_nnz_for_numa
        ));
    }

    (SpmvPlanFormat::Csr, reasons)
}

fn forced_bsr_format(
    candidate: Option<BsrPlanAnalysis>,
    format: SpmvPlanFormat,
    reasons: &mut Vec<String>,
) -> (SpmvPlanFormat, Vec<String>) {
    match candidate {
        Some(analysis) => {
            reasons.push(format!(
                "forced BSR{} by config: block_fill_ratio={:.3}, saved_bytes/apply={}",
                analysis.block_size, analysis.block_fill_ratio, analysis.saved_bytes_per_apply
            ));
            (format, reasons.clone())
        }
        None => {
            reasons.push(format!(
                "{format:?} requested, but matrix dimensions or block storage are not suitable"
            ));
            (SpmvPlanFormat::Csr, reasons.clone())
        }
    }
}

fn best_bsr_candidate(analysis: &SpmvAnalysis) -> Option<(SpmvPlanFormat, BsrPlanAnalysis)> {
    [
        (SpmvPlanFormat::Bsr2, analysis.bsr2),
        (SpmvPlanFormat::Bsr3, analysis.bsr3),
        (SpmvPlanFormat::Bsr4, analysis.bsr4),
    ]
    .into_iter()
    .filter_map(|(format, bsr)| bsr.map(|bsr| (format, bsr)))
    .max_by_key(|(_, bsr)| bsr.saved_bytes_per_apply)
}

fn analyze_csr<T: ComplexField>(matrix: &CsrMatrix<T>, config: &SpmvPlanConfig) -> SpmvAnalysis {
    let nnz = matrix.nnz();
    let mut min_row_nnz = usize::MAX;
    let mut max_row_nnz = 0usize;
    let mut empty_rows = 0usize;
    let mut sum = 0usize;
    let mut sum_sq = 0f64;

    for row in 0..matrix.num_rows {
        let row_nnz = matrix.row_ptrs[row + 1] - matrix.row_ptrs[row];
        min_row_nnz = min_row_nnz.min(row_nnz);
        max_row_nnz = max_row_nnz.max(row_nnz);
        if row_nnz == 0 {
            empty_rows += 1;
        }
        sum += row_nnz;
        sum_sq += (row_nnz as f64) * (row_nnz as f64);
    }

    if matrix.num_rows == 0 {
        min_row_nnz = 0;
    }

    let mean = if matrix.num_rows == 0 {
        0.0
    } else {
        sum as f64 / matrix.num_rows as f64
    };
    let variance = if matrix.num_rows == 0 {
        0.0
    } else {
        (sum_sq / matrix.num_rows as f64) - mean * mean
    };
    let max_col_index = matrix.col_indices.iter().copied().max();
    let can_use_u32_indices = indices_fit_u32(matrix);
    let value_bytes = std::mem::size_of::<T>();
    let csr_usize_bytes = bytes_csr_spmv(
        matrix.num_rows,
        matrix.num_cols,
        nnz,
        value_bytes,
        std::mem::size_of::<usize>(),
    );
    let csr_u32_bytes = can_use_u32_indices.then(|| {
        bytes_csr_spmv(
            matrix.num_rows,
            matrix.num_cols,
            nnz,
            value_bytes,
            std::mem::size_of::<u32>(),
        )
    });
    let bsr2 = analyze_bsr(matrix, 2, csr_usize_bytes);
    let bsr3 = analyze_bsr(matrix, 3, csr_usize_bytes);
    let bsr4 = analyze_bsr(matrix, 4, csr_usize_bytes);
    let row_bucket = analyze_row_bucket(matrix, csr_usize_bytes);
    let sell_c_sigma = analyze_sell_c_sigma(
        matrix,
        config.sell_c_sigma_chunk_size,
        config.sell_c_sigma_window,
    );

    SpmvAnalysis {
        nrows: matrix.num_rows,
        ncols: matrix.num_cols,
        nnz,
        empty_rows,
        min_row_nnz,
        max_row_nnz,
        mean_row_nnz: mean,
        row_nnz_stddev: variance.max(0.0).sqrt(),
        row_nnz_cv: if mean > 0.0 {
            variance.max(0.0).sqrt() / mean
        } else {
            0.0
        },
        max_col_index,
        can_use_u32_indices,
        csr_usize_bytes_per_apply: csr_usize_bytes,
        csr_u32_bytes_per_apply: csr_u32_bytes,
        saved_index_bytes_per_apply: csr_u32_bytes.map(|bytes| csr_usize_bytes - bytes),
        row_bucket,
        sell_c_sigma,
        bsr2,
        bsr3,
        bsr4,
    }
}

fn chosen_index_bytes(format: SpmvPlanFormat) -> usize {
    match format {
        SpmvPlanFormat::CompactCsrU32 => std::mem::size_of::<u32>(),
        SpmvPlanFormat::Csr
        | SpmvPlanFormat::NumaCsr
        | SpmvPlanFormat::RowBucketedCsr
        | SpmvPlanFormat::SellCSigma
        | SpmvPlanFormat::Bsr2
        | SpmvPlanFormat::Bsr3
        | SpmvPlanFormat::Bsr4 => std::mem::size_of::<usize>(),
    }
}

fn chosen_bytes_per_apply(format: SpmvPlanFormat, analysis: &SpmvAnalysis) -> u64 {
    match format {
        SpmvPlanFormat::CompactCsrU32 => analysis
            .csr_u32_bytes_per_apply
            .unwrap_or(analysis.csr_usize_bytes_per_apply),
        SpmvPlanFormat::Csr | SpmvPlanFormat::NumaCsr => analysis.csr_usize_bytes_per_apply,
        SpmvPlanFormat::RowBucketedCsr => analysis
            .row_bucket
            .map(|row_bucket| row_bucket.bytes_per_apply)
            .unwrap_or(analysis.csr_usize_bytes_per_apply),
        SpmvPlanFormat::SellCSigma => analysis
            .sell_c_sigma
            .map(|sell| sell.bytes_per_apply)
            .unwrap_or(analysis.csr_usize_bytes_per_apply),
        SpmvPlanFormat::Bsr2 => analysis
            .bsr2
            .map(|bsr| bsr.bytes_per_apply)
            .unwrap_or(analysis.csr_usize_bytes_per_apply),
        SpmvPlanFormat::Bsr3 => analysis
            .bsr3
            .map(|bsr| bsr.bytes_per_apply)
            .unwrap_or(analysis.csr_usize_bytes_per_apply),
        SpmvPlanFormat::Bsr4 => analysis
            .bsr4
            .map(|bsr| bsr.bytes_per_apply)
            .unwrap_or(analysis.csr_usize_bytes_per_apply),
    }
}

fn analyze_row_bucket<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    csr_bytes_per_apply: u64,
) -> Option<RowBucketPlanAnalysis> {
    if matrix.num_rows == 0 {
        return None;
    }

    let mut row_lengths = (0..matrix.num_rows)
        .map(|row| matrix.row_ptrs[row + 1] - matrix.row_ptrs[row])
        .collect::<Vec<_>>();
    row_lengths.sort_unstable();

    let mut bucket_count = 0usize;
    let mut largest_bucket_rows = 0usize;
    let mut singleton_buckets = 0usize;
    let mut idx = 0usize;
    while idx < row_lengths.len() {
        let len = row_lengths[idx];
        let start = idx;
        while idx < row_lengths.len() && row_lengths[idx] == len {
            idx += 1;
        }
        let bucket_rows = idx - start;
        bucket_count += 1;
        largest_bucket_rows = largest_bucket_rows.max(bucket_rows);
        if bucket_rows == 1 {
            singleton_buckets += 1;
        }
    }

    let row_order_bytes = (matrix.num_rows * std::mem::size_of::<usize>()) as u64;
    Some(RowBucketPlanAnalysis {
        bucket_count,
        largest_bucket_rows,
        singleton_buckets,
        bytes_per_apply: csr_bytes_per_apply.saturating_add(row_order_bytes),
        row_order_bytes,
    })
}

fn analyze_sell_c_sigma<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    chunk_size: usize,
    sigma: usize,
) -> Option<SellCSigmaPlanAnalysis> {
    if matrix.num_rows == 0 {
        return None;
    }

    let chunk_size = chunk_size.max(1);
    let sigma = sigma.max(chunk_size);
    let row_order = sell_c_sigma_row_order(matrix, chunk_size, sigma);
    let mut chunks = 0usize;
    let mut padded_nnz = 0usize;

    for chunk_start in (0..row_order.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(row_order.len());
        let rows = &row_order[chunk_start..chunk_end];
        let height = rows.len();
        let width = rows
            .iter()
            .map(|&row| matrix.row_ptrs[row + 1] - matrix.row_ptrs[row])
            .max()
            .unwrap_or(0);
        chunks += 1;
        padded_nnz = padded_nnz.saturating_add(height.saturating_mul(width));
    }

    let nnz = matrix.nnz();
    let fill_ratio = if padded_nnz == 0 {
        1.0
    } else {
        nnz as f64 / padded_nnz as f64
    };
    let bytes_per_apply = bytes_sell_c_sigma_spmv(
        matrix.num_rows,
        matrix.num_cols,
        padded_nnz,
        std::mem::size_of::<T>(),
        std::mem::size_of::<usize>(),
    );

    Some(SellCSigmaPlanAnalysis {
        chunk_size,
        sigma,
        chunks,
        padded_nnz,
        fill_ratio,
        bytes_per_apply,
        padding_overhead_nnz: padded_nnz.saturating_sub(nnz),
    })
}

fn sell_c_sigma_row_order<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    chunk_size: usize,
    sigma: usize,
) -> Vec<usize> {
    let chunk_size = chunk_size.max(1);
    let sigma = sigma.max(chunk_size);
    let mut row_order = Vec::with_capacity(matrix.num_rows);

    for window_start in (0..matrix.num_rows).step_by(sigma) {
        let window_end = (window_start + sigma).min(matrix.num_rows);
        let mut rows = (window_start..window_end)
            .map(|row| {
                let row_nnz = matrix.row_ptrs[row + 1] - matrix.row_ptrs[row];
                (row, row_nnz)
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|(row_a, nnz_a), (row_b, nnz_b)| {
            nnz_b.cmp(nnz_a).then_with(|| row_a.cmp(row_b))
        });
        row_order.extend(rows.into_iter().map(|(row, _)| row));
    }

    row_order
}

fn analyze_bsr<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    block_size: usize,
    csr_bytes_per_apply: u64,
) -> Option<BsrPlanAnalysis> {
    if block_size == 0
        || matrix.num_rows == 0
        || !matrix.num_rows.is_multiple_of(block_size)
        || !matrix.num_cols.is_multiple_of(block_size)
    {
        return None;
    }

    let block_rows = matrix.num_rows / block_size;
    let block_cols = matrix.num_cols / block_size;
    let mut nnz_blocks = 0usize;
    let mut block_cols_in_row = Vec::new();

    for block_row in 0..block_rows {
        block_cols_in_row.clear();
        for local_row in 0..block_size {
            let row = block_row * block_size + local_row;
            for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                block_cols_in_row.push(matrix.col_indices[idx] / block_size);
            }
        }
        block_cols_in_row.sort_unstable();
        block_cols_in_row.dedup();
        nnz_blocks += block_cols_in_row.len();
    }

    if nnz_blocks == 0 {
        return None;
    }

    let padded_scalar_nnz = nnz_blocks.saturating_mul(block_size * block_size);
    let bytes_per_apply = bytes_bsr_spmv(
        matrix.num_rows,
        matrix.num_cols,
        block_rows,
        nnz_blocks,
        padded_scalar_nnz,
        std::mem::size_of::<T>(),
        std::mem::size_of::<usize>(),
    );
    let saved_bytes_per_apply = csr_bytes_per_apply.saturating_sub(bytes_per_apply);
    if saved_bytes_per_apply == 0 {
        return None;
    }

    Some(BsrPlanAnalysis {
        block_size,
        block_rows,
        block_cols,
        nnz_blocks,
        padded_scalar_nnz,
        block_fill_ratio: matrix.nnz() as f64 / padded_scalar_nnz as f64,
        bytes_per_apply,
        saved_bytes_per_apply,
    })
}

fn indices_fit_u32<T: ComplexField>(matrix: &CsrMatrix<T>) -> bool {
    matrix.num_rows <= u32::MAX as usize
        && matrix.num_cols <= u32::MAX as usize
        && matrix.nnz() <= u32::MAX as usize
        && matrix
            .col_indices
            .iter()
            .all(|&col| col <= u32::MAX as usize)
}

fn bytes_csr_spmv(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    value_bytes: usize,
    index_bytes: usize,
) -> u64 {
    (value_bytes * nnz
        + index_bytes * nnz
        + index_bytes * (nrows + 1)
        + value_bytes * ncols
        + value_bytes * nrows) as u64
}

fn bytes_bsr_spmv(
    nrows: usize,
    ncols: usize,
    block_rows: usize,
    nnz_blocks: usize,
    padded_scalar_nnz: usize,
    value_bytes: usize,
    index_bytes: usize,
) -> u64 {
    (value_bytes * padded_scalar_nnz
        + index_bytes * nnz_blocks
        + index_bytes * (block_rows + 1)
        + value_bytes * ncols
        + value_bytes * nrows) as u64
}

fn bytes_sell_c_sigma_spmv(
    nrows: usize,
    ncols: usize,
    padded_nnz: usize,
    value_bytes: usize,
    index_bytes: usize,
) -> u64 {
    (value_bytes * padded_nnz
        + index_bytes * padded_nnz
        + index_bytes * nrows
        + value_bytes * ncols
        + value_bytes * nrows) as u64
}

impl<T: ComplexField> CsrMatrix<T> {
    /// Build a checked SpMV plan from this CSR matrix.
    pub fn try_spmv_plan(&self, config: SpmvPlanConfig) -> Result<SpmvPlan<T>, SolverError> {
        SpmvPlan::try_from_csr(self, config)
    }

    pub fn spmv_plan(&self, config: SpmvPlanConfig) -> SpmvPlan<T> {
        SpmvPlan::from_csr(self, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::{Complex32, Complex64};
    use num_traits::ToPrimitive;
    use sonium_solvers_core::dense::DenseMatrix;

    #[test]
    fn analysis_detects_u32_savings() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 2.0, 3.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let plan = SpmvPlan::from_csr(
            &csr,
            SpmvPlanConfig {
                preference: SpmvPlanPreference::Csr,
                ..Default::default()
            },
        );
        let analysis = &plan.report().analysis;

        assert!(analysis.can_use_u32_indices);
        assert!(analysis.saved_index_bytes_per_apply.unwrap() > 0);
        assert_eq!(analysis.max_row_nnz, 2);
        assert_eq!(analysis.empty_rows, 0);
        assert!(analysis.row_nnz_cv >= 0.0);
        assert_eq!(
            plan.report().chosen_bytes_per_apply,
            analysis.csr_usize_bytes_per_apply
        );
    }

    #[test]
    fn compact_plan_matches_csr_f64() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 5.0],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let plan = SpmvPlan::from_csr(
            &csr,
            SpmvPlanConfig {
                preference: SpmvPlanPreference::CompactCsrU32,
                ..Default::default()
            },
        );
        let x = vec![1.0_f64, 2.0, 3.0];
        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);

        assert_eq!(plan.format(), SpmvPlanFormat::CompactCsrU32);
        assert_eq!(plan.report().chosen_index_bytes, std::mem::size_of::<u32>());
        assert!(plan.report().estimated_saved_bytes_per_apply > 0);
        for i in 0..y_csr.len() {
            assert_relative_eq!(y_plan[i], y_csr[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn compact_plan_matches_csr_complex() {
        let c = |re, im| Complex64::new(re, im);
        let dense = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![c(2.0, 1.0), c(0.0, 0.0), c(1.0, -1.0), c(3.0, 0.5)],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::CompactCsrU32,
            ..Default::default()
        });
        let x = vec![c(1.0, 0.5), c(-2.0, 1.0)];
        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);

        for i in 0..y_csr.len() {
            assert_relative_eq!((y_plan[i] - y_csr[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn compact_checked_execution_rejects_mutated_storage_before_indexing() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, 1.0, 3.0, 4.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut compact = CompactCsrMatrix::try_from_csr(&csr).expect("compact conversion");
        compact.col_indices[0] = compact.num_cols as u32 + 1;

        let forward = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compact.try_apply(&[1.0, 2.0])
        }))
        .expect("checked compact apply must not panic");
        assert!(matches!(
            forward,
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "compact CSR SpMV"
        ));

        let transpose = compact
            .try_apply_transpose(&[1.0, 2.0])
            .expect_err("invalid compact CSR transpose must be rejected");
        assert!(matches!(
            transpose,
            SolverError::OperationFailed { operation, .. }
                if operation == "compact CSR transpose SpMV"
        ));

        let hermitian = compact
            .try_apply_hermitian(&[1.0, 2.0])
            .expect_err("invalid compact CSR Hermitian product must be rejected");
        assert!(matches!(
            hermitian,
            SolverError::OperationFailed { operation, .. }
                if operation == "compact CSR Hermitian SpMV"
        ));
    }

    #[test]
    fn compact_checked_execution_rejects_nonfinite_values_and_dimensions() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut compact = CompactCsrMatrix::try_from_csr(&csr).expect("compact conversion");
        compact.values[0] = f64::NAN;

        let nonfinite = compact
            .try_apply(&[1.0, 2.0])
            .expect_err("non-finite compact values must be rejected");
        assert!(matches!(
            nonfinite,
            SolverError::OperationFailed { operation, .. }
                if operation == "compact CSR SpMV"
        ));

        compact.values[0] = 1.0;
        let dimension = compact
            .try_apply(&[1.0])
            .expect_err("wrong compact CSR input dimension must be rejected");
        assert!(matches!(
            dimension,
            SolverError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn row_bucketed_checked_execution_rejects_mutated_row_order() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, 1.0, 3.0, 4.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut bucketed = RowBucketedCsrMatrix::try_from_csr(&csr).expect("row bucketing");
        bucketed.row_order[1] = bucketed.row_order[0];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bucketed.try_apply(&[1.0, 2.0])
        }))
        .expect("checked row-bucketed apply must not panic");
        assert!(matches!(
            result,
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "row-bucketed CSR SpMV"
        ));
    }

    #[test]
    fn sell_checked_execution_rejects_mutated_row_order() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![2.0_f64, 1.0, 3.0, 4.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let mut sell = SellCSigmaMatrix::try_from_csr(&csr, 1, 1).expect("SELL conversion");
        sell.row_order[1] = sell.row_order[0];

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sell.try_apply(&[1.0, 2.0])))
                .expect("checked SELL apply must not panic");
        assert!(matches!(
            result,
            Err(SolverError::OperationFailed { operation, .. })
                if operation == "SELL-C-sigma SpMV"
        ));
    }

    #[test]
    fn prebaked_transpose_matches_csr_transpose() {
        let dense = DenseMatrix::from_row_major_vec(
            3,
            4,
            vec![
                1.0_f64, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 7.0,
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            prebake_transpose: true,
            ..Default::default()
        });
        let x = vec![1.0_f64, -2.0, 0.5];
        let mut y_plan = vec![0.0_f64; csr.num_cols];
        plan.apply_transpose_into(&x, &mut y_plan);
        let y_csr = csr.matvec_transpose(&x);

        assert!(plan.report().transpose_prebaked);
        for i in 0..y_csr.len() {
            assert_relative_eq!(y_plan[i], y_csr[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn prebaked_hermitian_matches_csr_hermitian_complex() {
        let c = |re, im| Complex64::new(re, im);
        let dense = DenseMatrix::from_row_major_vec(
            2,
            3,
            vec![
                c(1.0, 2.0),
                c(0.0, 0.0),
                c(3.0, -1.0),
                c(-2.0, 0.5),
                c(4.0, 0.0),
                c(0.0, 0.0),
            ],
        );
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            prebake_hermitian: true,
            ..Default::default()
        });
        let x = vec![c(1.0, -1.0), c(0.5, 2.0)];
        let y_plan = plan.apply_hermitian(&x);
        let y_csr = csr.matvec_hermitian(&x);

        assert!(plan.report().hermitian_prebaked);
        assert!(!plan.report().transpose_prebaked);
        assert!(plan.report().summary().contains("hermitian_prebaked=true"));
        for i in 0..y_csr.len() {
            assert_relative_eq!((y_plan[i] - y_csr[i]).norm(), 0.0, epsilon = 1e-12);
        }
    }

    fn block_tridiagonal_2x2(blocks: usize) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for block_row in 0..blocks {
            let first_block_col = block_row.saturating_sub(1);
            let last_block_col = (block_row + 1).min(blocks - 1);
            for block_col in first_block_col..=last_block_col {
                for local_row in 0..2 {
                    for local_col in 0..2 {
                        let row = block_row * 2 + local_row;
                        let col = block_col * 2 + local_col;
                        let value = 10.0
                            + row as f64
                            + 0.25 * col as f64
                            + if block_row == block_col { 3.0 } else { 0.0 };
                        triplets.push((row, col, value));
                    }
                }
            }
        }
        CsrMatrix::from_triplets(blocks * 2, blocks * 2, triplets)
    }

    #[test]
    fn forced_bsr_plan_matches_csr_and_transpose() {
        let csr = block_tridiagonal_2x2(4);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::Bsr2,
            prebake_transpose: true,
            ..Default::default()
        });
        let x = (0..csr.num_cols)
            .map(|i| 1.0_f64 + i as f64 * 0.25)
            .collect::<Vec<_>>();
        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);

        assert_eq!(plan.format(), SpmvPlanFormat::Bsr2);
        assert!(!plan.report().transpose_prebaked);
        assert_eq!(plan.report().analysis.bsr2.unwrap().block_fill_ratio, 1.0);
        for i in 0..y_csr.len() {
            assert_relative_eq!(y_plan[i], y_csr[i], epsilon = 1e-12);
        }

        let mut y_transpose_plan = vec![0.0_f64; csr.num_cols];
        plan.apply_transpose_into(&x, &mut y_transpose_plan);
        let y_transpose_csr = csr.matvec_transpose(&x);
        for i in 0..y_transpose_csr.len() {
            assert_relative_eq!(y_transpose_plan[i], y_transpose_csr[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn auto_chooses_bsr_for_dense_system_blocks() {
        let csr = block_tridiagonal_2x2(64);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 1,
            min_reuse_for_bsr: 1,
            min_saved_bytes_for_bsr: 1,
            min_bsr_block_fill_ratio: 1.0,
            min_nnz_for_numa: usize::MAX,
            ..Default::default()
        });

        assert_eq!(plan.format(), SpmvPlanFormat::Bsr2);
        assert!(
            plan.report().chosen_bytes_per_apply < plan.report().analysis.csr_usize_bytes_per_apply
        );
        assert!(
            plan.report()
                .reasons
                .iter()
                .any(|reason| reason.contains("auto selected BSR2"))
        );
    }

    fn irregular_rows_matrix(rows: usize, cols: usize) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for row in 0..rows {
            let row_nnz = if row % 8 == 0 { cols / 2 } else { 1 };
            for offset in 0..row_nnz {
                let col = (row + offset * 3) % cols;
                let value = 1.0 + row as f64 * 0.01 + offset as f64 * 0.001;
                triplets.push((row, col, value));
            }
        }
        CsrMatrix::from_triplets(rows, cols, triplets)
    }

    fn mostly_uniform_band_matrix(rows: usize, cols: usize, bandwidth: usize) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for row in 0..rows {
            let start = row.saturating_sub(bandwidth / 2);
            let end = (start + bandwidth).min(cols);
            for col in start..end {
                let value = if row == col { 4.0 } else { 0.25 };
                triplets.push((row, col, value));
            }
        }
        CsrMatrix::from_triplets(rows, cols, triplets)
    }

    fn complex_rectangular_band_matrix(
        rows: usize,
        cols: usize,
        bandwidth: usize,
    ) -> CsrMatrix<Complex64> {
        let mut triplets = Vec::new();
        for row in 0..rows {
            let start = row.saturating_sub(bandwidth / 2);
            let end = (start + bandwidth).min(cols);
            for col in start..end {
                let re = 1.0 + row as f64 * 0.25 + col as f64 * 0.125;
                let im = if row == col { 0.5 } else { -0.25 };
                triplets.push((row, col, Complex64::new(re, im)));
            }
        }
        CsrMatrix::from_triplets(rows, cols, triplets)
    }

    #[test]
    fn forced_sell_c_sigma_plan_matches_csr() {
        let csr = mostly_uniform_band_matrix(31, 31, 7);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::SellCSigma,
            sell_c_sigma_chunk_size: 4,
            sell_c_sigma_window: 16,
            ..Default::default()
        });
        let x = (0..csr.num_cols)
            .map(|i| 1.0_f64 + i as f64 * 0.125)
            .collect::<Vec<_>>();
        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);

        assert_eq!(plan.format(), SpmvPlanFormat::SellCSigma);
        assert!(plan.report().analysis.sell_c_sigma.is_some());
        assert!(plan.report().summary().contains("sell_c_sigma:C=4"));
        for i in 0..y_csr.len() {
            assert_relative_eq!(y_plan[i], y_csr[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn compact_large_plan_matches_csr_through_apply_into() {
        let csr = mostly_uniform_band_matrix(2048, 2048, 31);
        assert!(csr.nnz() >= COMPACT_CSR_PAR_NNZ_THRESHOLD);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::CompactCsrU32,
            ..Default::default()
        });
        let x = (0..csr.num_cols)
            .map(|i| 1.0_f64 + i as f64 * 0.00025)
            .collect::<Vec<_>>();
        let mut y_plan = vec![0.0_f64; csr.num_rows];
        plan.apply_into(&x, &mut y_plan);
        let y_csr = csr.matvec(&x);

        assert_eq!(plan.format(), SpmvPlanFormat::CompactCsrU32);
        for i in 0..y_csr.len() {
            assert_relative_eq!(y_plan[i], y_csr[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn sell_c_sigma_prebakes_transpose_and_hermitian() {
        let csr = complex_rectangular_band_matrix(17, 19, 7);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::SellCSigma,
            prebake_transpose: true,
            prebake_hermitian: true,
            sell_c_sigma_chunk_size: 4,
            sell_c_sigma_window: 16,
            ..Default::default()
        });
        let x = (0..csr.num_rows)
            .map(|i| Complex64::new(1.0 + i as f64 * 0.125, -0.5 + i as f64 * 0.0625))
            .collect::<Vec<_>>();
        let mut y_transpose_plan = vec![Complex64::new(0.0, 0.0); csr.num_cols];
        plan.apply_transpose_into(&x, &mut y_transpose_plan);
        let y_transpose_csr = csr.matvec_transpose(&x);
        let y_hermitian_plan = plan.apply_hermitian(&x);
        let y_hermitian_csr = csr.matvec_hermitian(&x);

        assert_eq!(plan.format(), SpmvPlanFormat::SellCSigma);
        assert!(plan.report().transpose_prebaked);
        assert!(plan.report().hermitian_prebaked);
        for i in 0..y_transpose_csr.len() {
            assert_relative_eq!(
                (y_transpose_plan[i] - y_transpose_csr[i]).norm(),
                0.0,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                (y_hermitian_plan[i] - y_hermitian_csr[i]).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn auto_chooses_sell_c_sigma_for_reused_regular_rows() {
        let csr = mostly_uniform_band_matrix(128, 128, 9);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 16,
            sell_c_sigma_chunk_size: 8,
            sell_c_sigma_window: 32,
            min_reuse_for_sell_c_sigma: 2,
            min_nnz_for_sell_c_sigma: 1,
            min_sell_c_sigma_fill_ratio: 0.75,
            min_reuse_for_bsr: usize::MAX,
            min_nnz_for_numa: usize::MAX,
            ..Default::default()
        });

        assert_eq!(plan.format(), SpmvPlanFormat::SellCSigma);
        assert!(
            plan.report()
                .reasons
                .iter()
                .any(|reason| reason.contains("auto selected SELL-C-sigma"))
        );
    }

    #[test]
    fn forced_row_bucketed_plan_matches_csr() {
        let csr = irregular_rows_matrix(17, 19);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::RowBucketedCsr,
            ..Default::default()
        });
        let x = (0..csr.num_cols)
            .map(|i| 1.0_f64 + i as f64 * 0.125)
            .collect::<Vec<_>>();
        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);

        assert_eq!(plan.format(), SpmvPlanFormat::RowBucketedCsr);
        assert!(plan.report().analysis.row_bucket.is_some());
        assert!(plan.report().summary().contains("row_bucket:buckets="));
        for i in 0..y_csr.len() {
            assert_relative_eq!(y_plan[i], y_csr[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn auto_chooses_row_bucket_for_irregular_reused_matrix() {
        let csr = irregular_rows_matrix(128, 127);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 8,
            min_reuse_for_row_bucket: 2,
            min_nnz_for_row_bucket: 1,
            min_row_nnz_cv_for_row_bucket: 0.5,
            min_nnz_for_numa: usize::MAX,
            ..Default::default()
        });

        assert_eq!(plan.format(), SpmvPlanFormat::RowBucketedCsr);
        assert!(
            plan.report()
                .reasons
                .iter()
                .any(|reason| reason.contains("auto selected row-bucketed CSR"))
        );
    }

    #[test]
    fn auto_keeps_csr_for_low_reuse_small_matrix() {
        let dense = DenseMatrix::from_row_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0]);
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let plan = SpmvPlan::auto(&csr);

        assert_eq!(plan.format(), SpmvPlanFormat::Csr);
        assert!(
            plan.report()
                .reasons
                .iter()
                .any(|reason| reason.contains("kept CSR"))
        );
    }

    #[test]
    fn auto_chooses_compact_when_reuse_and_savings_are_high() {
        let n = 4096;
        let triplets = (0..n).map(|i| (i, i, 1.0_f64)).collect::<Vec<_>>();
        let csr = CsrMatrix::from_triplets(n, n, triplets);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            expected_reuse: 16,
            min_reuse_for_compact_indices: 2,
            min_saved_index_bytes_for_compact: 1,
            ..Default::default()
        });

        assert_eq!(plan.format(), SpmvPlanFormat::CompactCsrU32);
    }

    // -----------------------------------------------------------------
    // SIMD kernel parity: CompactCsrU32 and SELL-C-sigma vs CsrMatrix
    // -----------------------------------------------------------------

    /// Deterministic PRNG (xorshift64*) so the parity tests need no `rand`.
    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// Uniform-ish value in (-1, 1].
        fn unit(&mut self) -> f64 {
            (self.next() % 2000) as f64 / 1000.0 - 1.0
        }
    }

    /// Random sparse matrix with a deliberately uneven row-length
    /// distribution: every 7th row is empty, every 7th row is at maximum
    /// length, and the rest are uniform in `0..=max_row_nnz`.
    fn random_sparse_matrix<T: ComplexField>(
        rows: usize,
        cols: usize,
        max_row_nnz: usize,
        seed: u64,
        value: &dyn Fn(f64, f64) -> T,
    ) -> CsrMatrix<T> {
        let mut rng = XorShift64(seed.max(1));
        let max_row_nnz = max_row_nnz.min(cols);
        let mut triplets = Vec::new();
        for row in 0..rows {
            let row_nnz = match row % 7 {
                0 => 0,
                3 => max_row_nnz,
                _ => rng.below(max_row_nnz + 1),
            };
            for _ in 0..row_nnz {
                let col = rng.below(cols);
                triplets.push((row, col, value(rng.unit(), rng.unit())));
            }
        }
        CsrMatrix::from_triplets(rows, cols, triplets)
    }

    fn random_vector<T: ComplexField>(
        len: usize,
        seed: u64,
        value: &dyn Fn(f64, f64) -> T,
    ) -> Vec<T> {
        let mut rng = XorShift64(seed.max(1));
        (0..len).map(|_| value(rng.unit(), rng.unit())).collect()
    }

    /// Assert that the plan's matvec matches `CsrMatrix::matvec` within an
    /// fp-reassociation tolerance (relative to `max(1, |reference|)`).
    fn assert_matvec_parity<T: ComplexField>(
        csr: &CsrMatrix<T>,
        x: &[T],
        preference: SpmvPlanPreference,
        format: SpmvPlanFormat,
        tol: f64,
    ) {
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference,
            sell_c_sigma_chunk_size: 16,
            sell_c_sigma_window: 64,
            ..Default::default()
        });
        assert_eq!(plan.format(), format);
        let y_plan = plan.matvec(x);
        let y_csr = csr.matvec(x);
        assert_eq!(y_plan.len(), y_csr.len());
        for (i, (&y_plan_i, &y_csr_i)) in y_plan.iter().zip(y_csr.iter()).enumerate() {
            let diff = (y_plan_i - y_csr_i).norm().to_f64().unwrap();
            let scale = y_csr_i.norm().to_f64().unwrap().max(1.0);
            assert!(
                diff <= tol * scale,
                "row {i}: |{y_plan_i:?} - {y_csr_i:?}| = {diff} exceeds {tol} * {scale}"
            );
        }
    }

    #[test]
    fn compact_simd_matches_csr_random_f64() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 11u64),
            (257, 129, 33, 22),
            (1024, 300, 40, 33),
            // Exercises the rayon row-parallel path (nnz and rows above the
            // compact dispatch thresholds).
            (4096, 512, 40, 44),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, _| re);
            let x = random_vector(cols, seed + 1, &|re, _| re);
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::CompactCsrU32,
                SpmvPlanFormat::CompactCsrU32,
                1e-12,
            );
        }
    }

    #[test]
    fn compact_simd_matches_csr_random_f32() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 55u64),
            (513, 200, 24, 66),
            (4096, 512, 40, 77),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, _| re as f32);
            let x = random_vector(cols, seed + 1, &|re, _| re as f32);
            // f32 accumulators reassociate across SIMD lanes; tolerance is
            // scaled accordingly.
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::CompactCsrU32,
                SpmvPlanFormat::CompactCsrU32,
                1e-4,
            );
        }
    }

    #[test]
    fn compact_simd_matches_csr_random_complex64() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 88u64),
            (300, 250, 30, 99),
            (4096, 512, 40, 111),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, im| {
                Complex64::new(re, im)
            });
            let x = random_vector(cols, seed + 1, &|re, im| Complex64::new(re, im));
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::CompactCsrU32,
                SpmvPlanFormat::CompactCsrU32,
                1e-12,
            );
        }
    }

    #[test]
    fn compact_simd_matches_csr_random_complex32() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 121u64),
            (300, 250, 30, 131),
            (4096, 512, 40, 141),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, im| {
                Complex32::new(re as f32, im as f32)
            });
            let x = random_vector(cols, seed + 1, &|re, im| {
                Complex32::new(re as f32, im as f32)
            });
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::CompactCsrU32,
                SpmvPlanFormat::CompactCsrU32,
                1e-4,
            );
        }
    }

    #[test]
    fn sell_simd_matches_csr_random_f64() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 151u64),
            (257, 129, 33, 161),
            (1000, 300, 40, 171),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, _| re);
            let x = random_vector(cols, seed + 1, &|re, _| re);
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::SellCSigma,
                SpmvPlanFormat::SellCSigma,
                1e-12,
            );
        }
    }

    #[test]
    fn sell_simd_parallel_chunks_match_csr_f64() {
        // Padded nnz and chunk count above the SELL parallel thresholds, so
        // this exercises the rayon chunk-parallel path.
        let csr = random_sparse_matrix(4096, 512, 40, 181, &|re, _| re);
        let x = random_vector(512, 182, &|re, _| re);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::SellCSigma,
            sell_c_sigma_chunk_size: 16,
            sell_c_sigma_window: 64,
            ..Default::default()
        });
        let SpmvPlan::SellCSigma { matrix, .. } = &plan else {
            panic!("expected SELL-C-sigma plan");
        };
        assert!(matrix.values.len() >= par_nnz_threshold());
        assert!(matrix.chunk_heights.len() >= SELL_C_SIGMA_PAR_MIN_CHUNKS);

        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);
        for i in 0..y_csr.len() {
            let diff = (y_plan[i] - y_csr[i]).abs();
            assert!(
                diff <= 1e-12 * y_csr[i].abs().max(1.0),
                "row {i}: diff {diff}"
            );
        }
    }

    #[test]
    fn sell_simd_matches_csr_random_f32() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 191u64),
            (513, 200, 24, 201),
            (4096, 512, 40, 211),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, _| re as f32);
            let x = random_vector(cols, seed + 1, &|re, _| re as f32);
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::SellCSigma,
                SpmvPlanFormat::SellCSigma,
                1e-4,
            );
        }
    }

    #[test]
    fn sell_simd_matches_csr_random_complex64() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 221u64),
            (300, 250, 30, 231),
            (4096, 512, 40, 241),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, im| {
                Complex64::new(re, im)
            });
            let x = random_vector(cols, seed + 1, &|re, im| Complex64::new(re, im));
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::SellCSigma,
                SpmvPlanFormat::SellCSigma,
                1e-12,
            );
        }
    }

    #[test]
    fn sell_simd_matches_csr_random_complex32() {
        for (rows, cols, max_row_nnz, seed) in [
            (64usize, 48usize, 12usize, 251u64),
            (300, 250, 30, 261),
            (4096, 512, 40, 271),
        ] {
            let csr = random_sparse_matrix(rows, cols, max_row_nnz, seed, &|re, im| {
                Complex32::new(re as f32, im as f32)
            });
            let x = random_vector(cols, seed + 1, &|re, im| {
                Complex32::new(re as f32, im as f32)
            });
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::SellCSigma,
                SpmvPlanFormat::SellCSigma,
                1e-4,
            );
        }
    }

    #[test]
    fn sell_simd_chunk_tail_cases_match_csr() {
        // Row counts around/below the C=16 chunk size exercise the SIMD lane
        // tail (height % lanes != 0) and single-row chunks.
        for rows in [1usize, 2, 3, 5, 15, 16, 17, 31, 33, 48] {
            let csr = random_sparse_matrix(rows, 37, 9, rows as u64 + 7, &|re, _| re);
            let x = random_vector(37, rows as u64 + 13, &|re, _| re);
            assert_matvec_parity(
                &csr,
                &x,
                SpmvPlanPreference::SellCSigma,
                SpmvPlanFormat::SellCSigma,
                1e-12,
            );
        }

        // All-empty matrix: every chunk has width 0.
        let csr: CsrMatrix<f64> = CsrMatrix::from_triplets(20, 10, Vec::new());
        let x = random_vector(10, 3, &|re, _| re);
        assert_matvec_parity(
            &csr,
            &x,
            SpmvPlanPreference::SellCSigma,
            SpmvPlanFormat::SellCSigma,
            1e-12,
        );

        // A band of fully empty rows between non-empty ones produces
        // width-0 chunks next to full SIMD-width chunks. Zero-length rows
        // must yield exact zeros in y.
        let mut triplets = Vec::new();
        for row in 0..40usize {
            if (16..32).contains(&row) {
                continue;
            }
            for col in 0..5 {
                triplets.push((row, col, 1.0 + row as f64 * 0.01 + col as f64));
            }
        }
        let csr = CsrMatrix::from_triplets(40, 5, triplets);
        let x = random_vector(5, 5, &|re, _| re);
        let plan = csr.spmv_plan(SpmvPlanConfig {
            preference: SpmvPlanPreference::SellCSigma,
            ..Default::default()
        });
        let y_plan = plan.matvec(&x);
        let y_csr = csr.matvec(&x);
        for (row, value) in y_plan.iter().enumerate().take(32).skip(16) {
            assert_eq!(*value, 0.0, "empty row {row} must be exactly zero");
        }
        for i in 0..y_csr.len() {
            let diff = (y_plan[i] - y_csr[i]).abs();
            assert!(
                diff <= 1e-12 * y_csr[i].abs().max(1.0),
                "row {i}: diff {diff}"
            );
        }
    }

    #[test]
    fn compact_simd_edge_cases_match_csr() {
        // Rows of length 0..=9 (SIMD remainders of the 4/8-wide kernels) and
        // an all-empty matrix.
        let mut triplets = Vec::new();
        for row in 0..10usize {
            for col in 0..row {
                triplets.push((row, col, 0.5 + row as f64 * 0.1 + col as f64 * 0.01));
            }
        }
        let csr = CsrMatrix::from_triplets(10, 10, triplets);
        let x = random_vector(10, 17, &|re, _| re);
        assert_matvec_parity(
            &csr,
            &x,
            SpmvPlanPreference::CompactCsrU32,
            SpmvPlanFormat::CompactCsrU32,
            1e-12,
        );

        let csr: CsrMatrix<f64> = CsrMatrix::from_triplets(8, 8, Vec::new());
        let x = random_vector(8, 19, &|re, _| re);
        assert_matvec_parity(
            &csr,
            &x,
            SpmvPlanPreference::CompactCsrU32,
            SpmvPlanFormat::CompactCsrU32,
            1e-12,
        );
    }

    #[test]
    fn checked_plan_dispatch_rejects_bad_inputs_across_formats() {
        let triplets = (0..8)
            .flat_map(|row| [(row, row, 3.0 + row as f64), (row, (row + 1) % 8, -0.25)])
            .collect::<Vec<_>>();
        let csr = CsrMatrix::from_triplets(8, 8, triplets);
        let x = (0..8)
            .map(|index| 1.0 + index as f64 * 0.125)
            .collect::<Vec<_>>();

        for preference in [
            SpmvPlanPreference::Csr,
            SpmvPlanPreference::CompactCsrU32,
            SpmvPlanPreference::NumaCsr,
            SpmvPlanPreference::RowBucketedCsr,
            SpmvPlanPreference::SellCSigma,
            SpmvPlanPreference::Bsr2,
            SpmvPlanPreference::Bsr3,
            SpmvPlanPreference::Bsr4,
        ] {
            let plan = SpmvPlan::try_from_csr(
                &csr,
                SpmvPlanConfig {
                    preference,
                    numa_partitions: Some(2),
                    sell_c_sigma_chunk_size: 4,
                    sell_c_sigma_window: 8,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{preference:?} plan construction failed: {error}"));

            let forward = plan
                .try_apply(&x)
                .unwrap_or_else(|error| panic!("{preference:?} checked forward failed: {error}"));
            let transpose = plan
                .try_apply_transpose(&x)
                .unwrap_or_else(|error| panic!("{preference:?} checked transpose failed: {error}"));
            let hermitian = plan
                .try_apply_hermitian(&x)
                .unwrap_or_else(|error| panic!("{preference:?} checked Hermitian failed: {error}"));
            assert_eq!(forward.len(), 8);
            assert_eq!(transpose.len(), 8);
            assert_eq!(hermitian.len(), 8);

            assert!(matches!(
                plan.try_apply(&x[..7]),
                Err(SolverError::DimensionMismatch {
                    expected: 8,
                    got: 7
                })
            ));
            assert!(matches!(
                plan.try_apply(&[f64::NAN; 8]),
                Err(SolverError::OperationFailed { .. })
            ));
            assert!(matches!(
                plan.try_apply_transpose(&x[..7]),
                Err(SolverError::DimensionMismatch {
                    expected: 8,
                    got: 7
                })
            ));
            assert!(matches!(
                plan.try_apply_hermitian(&[f64::INFINITY; 8]),
                Err(SolverError::OperationFailed { .. })
            ));
        }
    }
}
