//! Lightweight roofline models for compute backend benchmarks.
//!
//! These helpers describe the bytes/flops a benchmark intends to measure.
//! They deliberately do not time anything themselves; Criterion owns timing,
//! while this module keeps operation accounting consistent across benches.

use std::mem::size_of;

use crate::backend::CsrRowBucketStats;
use crate::scalar::{BufferElement, DeviceScalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfModel {
    pub bytes: u64,
    pub flops: u64,
}

impl PerfModel {
    pub const fn new(bytes: u64, flops: u64) -> Self {
        Self { bytes, flops }
    }

    pub fn arithmetic_intensity(self) -> f64 {
        if self.bytes == 0 {
            0.0
        } else {
            self.flops as f64 / self.bytes as f64
        }
    }
}

/// Convert shape arithmetic into the release-facing accounting domain without
/// allowing a user-controlled benchmark size to overflow an intermediate
/// `usize` multiplication or panic in a debug build.
#[inline]
fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[inline]
fn product2(lhs: usize, rhs: usize) -> u64 {
    count(lhs).saturating_mul(count(rhs))
}

#[inline]
fn product3(lhs: usize, middle: usize, rhs: usize) -> u64 {
    product2(lhs, middle).saturating_mul(count(rhs))
}

#[inline]
fn bytes(elements: usize, element_size: usize) -> u64 {
    product2(elements, element_size)
}

#[inline]
fn flops(elements: usize, per_element: u64) -> u64 {
    count(elements).saturating_mul(per_element)
}

pub fn copy_h2d_model<T: BufferElement>(n: usize) -> PerfModel {
    PerfModel::new(bytes(n, size_of::<T>()), 0)
}

pub fn copy_d2d_model<T: BufferElement>(n: usize) -> PerfModel {
    PerfModel::new(bytes(n, size_of::<T>()).saturating_mul(2), 0)
}

pub fn scale_model<T: DeviceScalar>(n: usize) -> PerfModel {
    PerfModel::new(
        bytes(n, size_of::<T>()).saturating_mul(2),
        flops(n, scale_flops_per_elem::<T>()),
    )
}

pub fn dot_model<T: DeviceScalar>(n: usize) -> PerfModel {
    let flops_per_elem = multiply_accumulate_flops::<T>();
    PerfModel::new(
        bytes(n, size_of::<T>()).saturating_mul(2),
        flops(n, flops_per_elem),
    )
}

pub fn axpy_model<T: DeviceScalar>(n: usize) -> PerfModel {
    let flops_per_elem = multiply_accumulate_flops::<T>();
    PerfModel::new(
        bytes(n, size_of::<T>()).saturating_mul(3),
        flops(n, flops_per_elem),
    )
}

pub fn nrm2_model<T: DeviceScalar>(n: usize) -> PerfModel {
    PerfModel::new(
        bytes(n, size_of::<T>()),
        flops(n, norm_flops_per_elem::<T>()),
    )
}

pub fn gemv_model<T: DeviceScalar>(m: usize, n: usize) -> PerfModel {
    let flops_per_entry = multiply_accumulate_flops::<T>();
    let entries = product2(m, n)
        .saturating_add(count(n))
        .saturating_add(count(m).saturating_mul(2));
    PerfModel::new(
        entries.saturating_mul(count(size_of::<T>())),
        product2(m, n).saturating_mul(flops_per_entry),
    )
}

pub fn gemm_model<T: DeviceScalar>(m: usize, n: usize, k: usize) -> PerfModel {
    let flops_per_entry = multiply_accumulate_flops::<T>();
    let entries = product2(m, k)
        .saturating_add(product2(k, n))
        .saturating_add(product2(m, n).saturating_mul(2));
    PerfModel::new(
        entries.saturating_mul(count(size_of::<T>())),
        product3(m, n, k).saturating_mul(flops_per_entry),
    )
}

pub fn ab_gemm_model<T: DeviceScalar>(m: usize, n: usize, k: usize) -> PerfModel {
    gemm_model::<T>(m, n, k)
}

pub fn gram_gemm_model<T: DeviceScalar>(rows: usize, cols: usize) -> PerfModel {
    // X^T X as a dense GEMM: (cols x rows) * (rows x cols).
    gemm_model::<T>(cols, cols, rows)
}

pub fn cov_gemm_model<T: DeviceScalar>(rows: usize, cols: usize) -> PerfModel {
    // This models the GEMM core of covariance assembly. Centering/mean
    // subtraction is intentionally separate from the dense product cost.
    gram_gemm_model::<T>(rows, cols)
}

pub fn csr_spmv_model<T: DeviceScalar>(stats: CsrRowBucketStats) -> PerfModel {
    csr_spmv_model_ncols::<T>(stats, stats.rows)
}

/// Accurate CSR SpMV roofline model: the source vector `x` has `n_cols`
/// elements (one per matrix column), not one per nonzero. The legacy
/// [`csr_spmv_model`] passes `stats.rows` as a square-matrix proxy; callers
/// with rectangular matrices must use this overload with the true column
/// count so arithmetic intensity is not skewed.
pub fn csr_spmv_model_ncols<T: DeviceScalar>(stats: CsrRowBucketStats, n_cols: usize) -> PerfModel {
    let value_bytes = bytes(stats.nnz, size_of::<T>());
    let col_bytes = bytes(stats.nnz, size_of::<i32>());
    let row_ptr_bytes = bytes(stats.rows.saturating_add(1), size_of::<i32>());
    let x_bytes = bytes(n_cols, size_of::<T>());
    let y_bytes = bytes(stats.rows, size_of::<T>());
    let flops_per_nnz = multiply_accumulate_flops::<T>();
    PerfModel::new(
        value_bytes
            .saturating_add(col_bytes)
            .saturating_add(row_ptr_bytes)
            .saturating_add(x_bytes)
            .saturating_add(y_bytes),
        flops(stats.nnz, flops_per_nnz),
    )
}

pub fn cast_model<Src: BufferElement, Dst: BufferElement>(n: usize) -> PerfModel {
    let element_bytes = size_of::<Src>().saturating_add(size_of::<Dst>());
    PerfModel::new(bytes(n, element_bytes), 0)
}

pub fn gather4_weighted_model<T: DeviceScalar>(rows: usize) -> PerfModel {
    let per_row = size_of::<i32>()
        .saturating_mul(4)
        .saturating_add(size_of::<T::Real>().saturating_mul(4))
        .saturating_add(size_of::<T>());
    let flops_per_row = if is_complex::<T>() { 32 } else { 8 };
    PerfModel::new(bytes(rows, per_row), flops(rows, flops_per_row))
}

pub fn bem_regular_pair_c32_model(pairs: usize) -> PerfModel {
    let input_bytes = bytes(pairs, size_of::<crate::bem::BemRegularPairIn>());
    let output_bytes = bytes(pairs, size_of::<crate::bem::BemRegularPairOut>());
    PerfModel::new(input_bytes.saturating_add(output_bytes), 0)
}

pub fn host_metadata() -> String {
    format!(
        "arch={} os={} family={} features={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::consts::FAMILY,
        active_features()
    )
}

fn active_features() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    if cfg!(feature = "bench-openblas") {
        features.push("bench-openblas");
    }

    if features.is_empty() {
        "default".to_string()
    } else {
        features.join(",")
    }
}

fn is_complex<T: DeviceScalar>() -> bool {
    size_of::<T>() == 2 * size_of::<T::Real>()
}

fn multiply_accumulate_flops<T: DeviceScalar>() -> u64 {
    if is_complex::<T>() { 8 } else { 2 }
}

fn scale_flops_per_elem<T: DeviceScalar>() -> u64 {
    if is_complex::<T>() { 6 } else { 1 }
}

fn norm_flops_per_elem<T: DeviceScalar>() -> u64 {
    if is_complex::<T>() { 4 } else { 2 }
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    use super::*;

    #[test]
    fn blas1_models_account_for_complex32_as_complex() {
        assert_eq!(scale_model::<Complex32>(10), PerfModel::new(160, 60));
        assert_eq!(dot_model::<Complex32>(10), PerfModel::new(160, 80));
        assert_eq!(axpy_model::<Complex32>(10), PerfModel::new(240, 80));
        assert_eq!(nrm2_model::<Complex32>(10), PerfModel::new(80, 40));
    }

    #[test]
    fn dense_models_use_complex_multiply_accumulate_counts() {
        assert_eq!(gemv_model::<f32>(4, 8), PerfModel::new(192, 64));
        assert_eq!(gemv_model::<Complex32>(4, 8), PerfModel::new(384, 256));
        assert_eq!(gemm_model::<f64>(4, 8, 16), PerfModel::new(2048, 1024));
        assert_eq!(
            gemm_model::<Complex64>(4, 8, 16),
            PerfModel::new(4096, 4096)
        );
    }

    #[test]
    fn flashlib_style_dense_models_are_gemm_shapes() {
        assert_eq!(ab_gemm_model::<f32>(4, 8, 16), gemm_model::<f32>(4, 8, 16));
        assert_eq!(gram_gemm_model::<f32>(16, 8), gemm_model::<f32>(8, 8, 16));
        assert_eq!(
            cov_gemm_model::<Complex32>(16, 8),
            gemm_model::<Complex32>(8, 8, 16)
        );
    }

    #[test]
    fn extreme_shapes_saturate_accounting_instead_of_wrapping() {
        let n = usize::MAX;
        assert_eq!(copy_h2d_model::<Complex64>(n).bytes, u64::MAX);
        assert_eq!(copy_d2d_model::<Complex64>(n).bytes, u64::MAX);
        assert_eq!(scale_model::<Complex64>(n).bytes, u64::MAX);
        assert_eq!(dot_model::<Complex64>(n).flops, u64::MAX);
        assert_eq!(gemv_model::<Complex64>(n, n).bytes, u64::MAX);
        assert_eq!(gemm_model::<Complex64>(n, n, n).flops, u64::MAX);
        assert_eq!(cast_model::<f32, Complex64>(n).bytes, u64::MAX);
        assert_eq!(gather4_weighted_model::<Complex64>(n).bytes, u64::MAX);
        assert_eq!(bem_regular_pair_c32_model(n).bytes, u64::MAX);
    }

    #[test]
    fn empty_models_have_zero_work_and_zero_intensity() {
        let model = gemm_model::<f64>(0, 0, 0);
        assert_eq!(model, PerfModel::new(0, 0));
        assert_eq!(model.arithmetic_intensity(), 0.0);
    }

    #[test]
    fn csr_spmv_model_uses_ncols_for_x_traffic() {
        // 1 row, 8 cols, 8 nonzeros: x traffic is 8 elements.
        let stats = CsrRowBucketStats {
            rows: 1,
            nnz: 8,
            empty_rows: 0,
            short_rows: 0,
            simd_rows: 0,
            cta_rows: 0,
            max_nnz_per_row: 8,
        };
        let model = csr_spmv_model_ncols::<f32>(stats, 8);
        // values 8*4 + cols 8*4 + row_ptr 2*4 + x 8*4 + y 1*4 = 108.
        assert_eq!(model.bytes, 108);
        // Rectangular: 2 rows, 8 cols, 4 nonzeros → x traffic is 8 elems.
        let stats = CsrRowBucketStats {
            rows: 2,
            nnz: 4,
            empty_rows: 0,
            short_rows: 0,
            simd_rows: 0,
            cta_rows: 0,
            max_nnz_per_row: 4,
        };
        let model = csr_spmv_model_ncols::<f32>(stats, 8);
        // values 4*4 + cols 4*4 + row_ptr 3*4 + x 8*4 + y 2*4 = 84.
        assert_eq!(model.bytes, 84);
    }
}
