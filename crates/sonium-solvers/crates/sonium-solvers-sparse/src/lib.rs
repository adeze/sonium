//! Sparse matrix structures (CSR, BSR formats)
//!
//! This module provides sparse matrix formats for efficient storage and
//! matrix-vector operations:
//!
//! - **CSR** (Compressed Sparse Row): general-purpose, SIMD-accelerated SpMV
//! - **BSR** (Block Sparse Row): for systems with fixed block size (Stokes, elasticity)

pub mod block_operator;
pub mod bsr;
mod csr;
pub mod numa;
pub mod operator;
pub mod partition;
pub mod spmv_plan;
pub mod triplet;

pub use block_operator::{BlockCsr, BlockLayout, BlockOperator, DenseBlockOperator, ZeroBlock};
pub use bsr::BsrMatrix;
pub use csr::{
    BlockedCsr, CsrBuilder, CsrMatrix, SpmvKernel, detected_spmv_features, selected_spmv_kernel,
};
pub use numa::{
    CsrPartition, NumaPartitionedCsr, NumaPartitioningReport, NumaPlacementStatus, NumaSpmvError,
    NumaStorageEstimate, PartitionStats, detect_numa_partitions,
};
pub use numa::{InputVectorPlacement, NumaSpmvPolicy};
pub use operator::{CompositeOperator, FnOperator, ScaledOperator, SumOperator};
pub use partition::try_extract_submatrix;
pub use sonium_compute::{DeviceBlock, DeviceBlockOperator, SquareDeviceBlock};
pub use spmv_plan::{
    BsrPlanAnalysis, CompactCsrMatrix, RowBucketPlanAnalysis, RowBucketedCsrMatrix,
    SellCSigmaMatrix, SellCSigmaPlanAnalysis, SpmvAnalysis, SpmvPlan, SpmvPlanConfig,
    SpmvPlanFormat, SpmvPlanPreference, SpmvPlanReport,
};
pub use triplet::TripletMatrix;
