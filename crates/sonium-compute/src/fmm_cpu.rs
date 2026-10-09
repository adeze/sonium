//! CPU helpers for FMM-style Complex64 kernels.
//!
//! These functions deliberately work on plain slices so `sonium-compute`
//! stays below the BEM and solver crates.  The common pattern is a small
//! row-major dense block whose input and/or output live in global DOF arrays:
//! gather indexed source entries, compute one row dot product, and scatter
//! the result into an indexed destination.

use num_complex::Complex64;

use crate::{BackendError, BackendResult};

/// One indexed row-major dense matvec operation in a batched
/// gather-dot-scatter dispatch.
#[derive(Debug, Clone, Copy)]
pub struct GatherDotScatterC64<'a> {
    /// Row-major matrix data.
    pub matrix: &'a [Complex64],
    /// Logical matrix row count to process.
    pub rows: usize,
    /// Logical matrix column count to process.
    pub cols: usize,
    /// Row stride in `matrix` elements.
    pub lda: usize,
    /// Source indices into the global input vector.
    pub src_indices: &'a [usize],
    /// Destination indices into the global output vector.
    pub dst_indices: &'a [usize],
}

#[inline]
fn c64_mul(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn c64_mul_add(acc: Complex64, a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(
        acc.re + a.re * b.re - a.im * b.im,
        acc.im + a.re * b.im + a.im * b.re,
    )
}

#[inline]
fn checked_dense_span(
    operation: &'static str,
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
) -> BackendResult<usize> {
    if lda < cols {
        return Err(BackendError::InvalidArgument {
            operation,
            reason: format!("leading dimension {lda} is smaller than column count {cols}"),
        });
    }
    let required = if rows == 0 {
        0
    } else {
        (rows - 1)
            .checked_mul(lda)
            .and_then(|offset| offset.checked_add(cols))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation,
                reason: "matrix span overflows usize".into(),
            })?
    };
    if matrix.len() < required {
        return Err(BackendError::InvalidArgument {
            operation,
            reason: format!(
                "matrix length {} is smaller than required span {required}",
                matrix.len()
            ),
        });
    }
    Ok(required)
}

fn assert_dense_shape(matrix: &[Complex64], rows: usize, cols: usize, lda: usize) -> bool {
    if let Err(error) = checked_dense_span("dense_matvec", matrix, rows, cols, lda) {
        eprintln!("dense matvec rejected input: {error}");
        return false;
    }
    true
}

/// Fused contiguous phase update:
/// `dst[p] += phase[p] * src[p]`.
#[inline]
pub fn try_phase_mul_accumulate_c64(
    dst: &mut [Complex64],
    phase: &[Complex64],
    src: &[Complex64],
) -> BackendResult<()> {
    if dst.len() != phase.len() {
        return Err(BackendError::InvalidArgument {
            operation: "phase_mul_accumulate_c64",
            reason: format!(
                "phase length {} does not match dst length {}",
                phase.len(),
                dst.len()
            ),
        });
    }
    if dst.len() != src.len() {
        return Err(BackendError::InvalidArgument {
            operation: "phase_mul_accumulate_c64",
            reason: format!(
                "src length {} does not match dst length {}",
                src.len(),
                dst.len()
            ),
        });
    }

    let (dst_chunks, dst_remainder) = dst.as_chunks_mut::<4>();
    let (phase_chunks, phase_remainder) = phase.as_chunks::<4>();
    let (src_chunks, src_remainder) = src.as_chunks::<4>();
    for ((dst4, phase4), src4) in dst_chunks
        .iter_mut()
        .zip(phase_chunks.iter())
        .zip(src_chunks.iter())
    {
        dst4[0] += c64_mul(phase4[0], src4[0]);
        dst4[1] += c64_mul(phase4[1], src4[1]);
        dst4[2] += c64_mul(phase4[2], src4[2]);
        dst4[3] += c64_mul(phase4[3], src4[3]);
    }
    for ((dst, phase), src) in dst_remainder
        .iter_mut()
        .zip(phase_remainder.iter())
        .zip(src_remainder.iter())
    {
        *dst += c64_mul(*phase, *src);
    }
    Ok(())
}

/// Compatibility wrapper for [`try_phase_mul_accumulate_c64`]. Invalid
/// lengths retain the historical panic behavior; checked callers should use
/// the fallible entry point.
#[inline]
pub fn phase_mul_accumulate_c64(dst: &mut [Complex64], phase: &[Complex64], src: &[Complex64]) {
    if let Err(error) = try_phase_mul_accumulate_c64(dst, phase, src) {
        eprintln!("phase_mul_accumulate_c64 rejected input: {error}");
    }
}

/// Compute `dst = A * src[src_indices]`.
///
/// Rows beyond `rows` are written as zero. Out-of-bounds source indices
/// contribute zero, matching the defensive BEM hot paths that skip stale or
/// pruned DOF references instead of panicking during a solve.
#[allow(clippy::too_many_arguments)]
pub fn indexed_dense_matvec_c64(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    src: &[Complex64],
    src_indices: &[usize],
    dst: &mut [Complex64],
) {
    let rows = rows.min(dst.len());
    let cols = cols.min(src_indices.len());
    if !assert_dense_shape(matrix, rows, cols, lda) {
        return;
    }
    dst.fill(Complex64::new(0.0, 0.0));

    for row_idx in 0..rows {
        let row = &matrix[row_idx * lda..row_idx * lda + cols];
        let mut acc = Complex64::new(0.0, 0.0);
        for (&a, &src_idx) in row.iter().zip(src_indices.iter().take(cols)) {
            if src_idx >= src.len() {
                continue;
            }
            acc = c64_mul_add(acc, a, src[src_idx]);
        }
        dst[row_idx] = acc;
    }
}

/// Fused indexed dense matvec:
/// `dst[dst_indices[i]] += Σ_j A[i,j] * src[src_indices[j]]`.
#[allow(clippy::too_many_arguments)]
pub fn indexed_dense_matvec_accumulate_c64(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    src: &[Complex64],
    src_indices: &[usize],
    dst: &mut [Complex64],
    dst_indices: &[usize],
) {
    let rows = rows.min(dst_indices.len());
    let cols = cols.min(src_indices.len());
    if !assert_dense_shape(matrix, rows, cols, lda) {
        return;
    }

    for row_idx in 0..rows {
        let global_dst = dst_indices[row_idx];
        if global_dst >= dst.len() {
            continue;
        }
        let row = &matrix[row_idx * lda..row_idx * lda + cols];
        let mut acc = Complex64::new(0.0, 0.0);
        for (&a, &src_idx) in row.iter().zip(src_indices.iter().take(cols)) {
            if src_idx >= src.len() {
                continue;
            }
            acc = c64_mul_add(acc, a, src[src_idx]);
        }
        dst[global_dst] += acc;
    }
}

/// Dense matvec with contiguous source and indexed scatter:
/// `dst[dst_indices[i]] += Σ_j A[i,j] * src[j]`.
#[allow(clippy::too_many_arguments)]
pub fn dense_matvec_scatter_accumulate_c64(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    src: &[Complex64],
    dst: &mut [Complex64],
    dst_indices: &[usize],
) {
    let rows = rows.min(dst_indices.len());
    let cols = cols.min(src.len());
    if !assert_dense_shape(matrix, rows, cols, lda) {
        return;
    }

    for row_idx in 0..rows {
        let global_dst = dst_indices[row_idx];
        if global_dst >= dst.len() {
            continue;
        }
        let row = &matrix[row_idx * lda..row_idx * lda + cols];
        let mut acc = Complex64::new(0.0, 0.0);
        for (&a, &x) in row.iter().zip(src.iter().take(cols)) {
            acc = c64_mul_add(acc, a, x);
        }
        dst[global_dst] += acc;
    }
}

/// Gather selected directions from a full expansion row into a compact cone row:
/// `dst[k] = src[indices[k]]`.
#[inline]
pub fn compact_indexed_gather_c64(src: &[Complex64], indices: &[usize], dst: &mut [Complex64]) {
    let len = indices.len().min(dst.len());
    for (dst_value, &src_idx) in dst.iter_mut().take(len).zip(indices.iter().take(len)) {
        if src_idx < src.len() {
            *dst_value = src[src_idx];
        }
    }
}

/// Scatter-add a compact cone row back into selected full expansion directions:
/// `dst[indices[k]] += src[k]`.
#[inline]
pub fn compact_indexed_scatter_accumulate_c64(
    src: &[Complex64],
    indices: &[usize],
    dst: &mut [Complex64],
) {
    let len = indices.len().min(src.len());
    for (&src_value, &dst_idx) in src.iter().take(len).zip(indices.iter().take(len)) {
        if dst_idx < dst.len() {
            dst[dst_idx] += src_value;
        }
    }
}

/// Batched gather-dot-scatter over leaf-cluster mappings.
///
/// Each batch item performs
/// `dst[dst_indices[i]] += Σ_j A[i,j] * src[src_indices[j]]`.
pub fn gather_dot_scatter_batch_c64(
    batches: &[GatherDotScatterC64<'_>],
    src: &[Complex64],
    dst: &mut [Complex64],
) {
    for batch in batches {
        indexed_dense_matvec_accumulate_c64(
            batch.matrix,
            batch.rows,
            batch.cols,
            batch.lda,
            src,
            batch.src_indices,
            dst,
            batch.dst_indices,
        );
    }
}

/// Checked indexed dense matvec. Unlike the compatibility wrapper, this
/// entry point requires the logical dimensions and all gather/scatter indices
/// to be valid before touching the destination.
#[allow(clippy::too_many_arguments)]
pub fn try_indexed_dense_matvec_c64(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    src: &[Complex64],
    src_indices: &[usize],
    dst: &mut [Complex64],
) -> BackendResult<()> {
    checked_dense_span("indexed_dense_matvec_c64", matrix, rows, cols, lda)?;
    if rows > dst.len() || cols > src_indices.len() {
        return Err(BackendError::InvalidArgument {
            operation: "indexed_dense_matvec_c64",
            reason: format!(
                "logical shape {rows}x{cols} exceeds destination {} or index list {}",
                dst.len(),
                src_indices.len()
            ),
        });
    }
    if let Some((position, &index)) = src_indices[..cols]
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= src.len())
    {
        return Err(BackendError::InvalidArgument {
            operation: "indexed_dense_matvec_c64",
            reason: format!("source index {index} at position {position} is out of bounds"),
        });
    }
    dst[..rows].fill(Complex64::new(0.0, 0.0));
    for row in 0..rows {
        let mut acc = Complex64::new(0.0, 0.0);
        for col in 0..cols {
            acc = c64_mul_add(acc, matrix[row * lda + col], src[src_indices[col]]);
        }
        dst[row] = acc;
    }
    Ok(())
}

/// Checked indexed dense matvec with indexed accumulation into `dst`.
#[allow(clippy::too_many_arguments)]
pub fn try_indexed_dense_matvec_accumulate_c64(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    src: &[Complex64],
    src_indices: &[usize],
    dst: &mut [Complex64],
    dst_indices: &[usize],
) -> BackendResult<()> {
    checked_dense_span(
        "indexed_dense_matvec_accumulate_c64",
        matrix,
        rows,
        cols,
        lda,
    )?;
    if rows > dst_indices.len() || cols > src_indices.len() {
        return Err(BackendError::InvalidArgument {
            operation: "indexed_dense_matvec_accumulate_c64",
            reason: "logical shape exceeds index-list lengths".into(),
        });
    }
    if let Some((position, &index)) = src_indices[..cols]
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= src.len())
    {
        return Err(BackendError::InvalidArgument {
            operation: "indexed_dense_matvec_accumulate_c64",
            reason: format!("source index {index} at position {position} is out of bounds"),
        });
    }
    if let Some((position, &index)) = dst_indices[..rows]
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= dst.len())
    {
        return Err(BackendError::InvalidArgument {
            operation: "indexed_dense_matvec_accumulate_c64",
            reason: format!("destination index {index} at position {position} is out of bounds"),
        });
    }
    for row in 0..rows {
        let mut acc = Complex64::new(0.0, 0.0);
        for col in 0..cols {
            acc = c64_mul_add(acc, matrix[row * lda + col], src[src_indices[col]]);
        }
        dst[dst_indices[row]] += acc;
    }
    Ok(())
}

/// Checked dense matvec with indexed scatter accumulation.
#[allow(clippy::too_many_arguments)]
pub fn try_dense_matvec_scatter_accumulate_c64(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    src: &[Complex64],
    dst: &mut [Complex64],
    dst_indices: &[usize],
) -> BackendResult<()> {
    checked_dense_span(
        "dense_matvec_scatter_accumulate_c64",
        matrix,
        rows,
        cols,
        lda,
    )?;
    if rows > dst_indices.len() || cols > src.len() {
        return Err(BackendError::InvalidArgument {
            operation: "dense_matvec_scatter_accumulate_c64",
            reason: "logical shape exceeds source or destination lengths".into(),
        });
    }
    if let Some((position, &index)) = dst_indices[..rows]
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= dst.len())
    {
        return Err(BackendError::InvalidArgument {
            operation: "dense_matvec_scatter_accumulate_c64",
            reason: format!("destination index {index} at position {position} is out of bounds"),
        });
    }
    for row in 0..rows {
        let mut acc = Complex64::new(0.0, 0.0);
        for col in 0..cols {
            acc = c64_mul_add(acc, matrix[row * lda + col], src[col]);
        }
        dst[dst_indices[row]] += acc;
    }
    Ok(())
}

/// Checked compact gather. Every index must be valid and the destination
/// length must match the index list exactly.
pub fn try_compact_indexed_gather_c64(
    src: &[Complex64],
    indices: &[usize],
    dst: &mut [Complex64],
) -> BackendResult<()> {
    if dst.len() != indices.len() {
        return Err(BackendError::InvalidArgument {
            operation: "compact_indexed_gather_c64",
            reason: "destination length must equal index-list length".into(),
        });
    }
    if let Some((position, &index)) = indices
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= src.len())
    {
        return Err(BackendError::InvalidArgument {
            operation: "compact_indexed_gather_c64",
            reason: format!("source index {index} at position {position} is out of bounds"),
        });
    }
    for (value, &index) in dst.iter_mut().zip(indices) {
        *value = src[index];
    }
    Ok(())
}

/// Checked compact scatter-add.
pub fn try_compact_indexed_scatter_accumulate_c64(
    src: &[Complex64],
    indices: &[usize],
    dst: &mut [Complex64],
) -> BackendResult<()> {
    if src.len() != indices.len() {
        return Err(BackendError::InvalidArgument {
            operation: "compact_indexed_scatter_accumulate_c64",
            reason: "source length must equal index-list length".into(),
        });
    }
    if let Some((position, &index)) = indices
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= dst.len())
    {
        return Err(BackendError::InvalidArgument {
            operation: "compact_indexed_scatter_accumulate_c64",
            reason: format!("destination index {index} at position {position} is out of bounds"),
        });
    }
    for (&value, &index) in src.iter().zip(indices) {
        dst[index] += value;
    }
    Ok(())
}

/// Checked batched gather-dot-scatter. Every batch is validated before any
/// destination entry is modified.
pub fn try_gather_dot_scatter_batch_c64(
    batches: &[GatherDotScatterC64<'_>],
    src: &[Complex64],
    dst: &mut [Complex64],
) -> BackendResult<()> {
    for batch in batches {
        checked_dense_span(
            "gather_dot_scatter_batch_c64",
            batch.matrix,
            batch.rows,
            batch.cols,
            batch.lda,
        )?;
        if batch.rows > batch.dst_indices.len() || batch.cols > batch.src_indices.len() {
            return Err(BackendError::InvalidArgument {
                operation: "gather_dot_scatter_batch_c64",
                reason: "logical shape exceeds batch index-list lengths".into(),
            });
        }
        if batch.src_indices[..batch.cols]
            .iter()
            .any(|&index| index >= src.len())
            || batch.dst_indices[..batch.rows]
                .iter()
                .any(|&index| index >= dst.len())
        {
            return Err(BackendError::InvalidArgument {
                operation: "gather_dot_scatter_batch_c64",
                reason: "batch gather or scatter index is out of bounds".into(),
            });
        }
    }
    for batch in batches {
        try_indexed_dense_matvec_accumulate_c64(
            batch.matrix,
            batch.rows,
            batch.cols,
            batch.lda,
            src,
            batch.src_indices,
            dst,
            batch.dst_indices,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn assert_c64_close(actual: Complex64, expected: Complex64) {
        assert_relative_eq!(actual.re, expected.re, max_relative = 1e-12);
        assert_relative_eq!(actual.im, expected.im, max_relative = 1e-12);
    }

    #[test]
    fn phase_mul_accumulate_matches_reference() {
        let phase = vec![
            c(1.0, 0.5),
            c(-0.25, 0.75),
            c(0.0, -2.0),
            c(3.0, 1.0),
            c(-1.0, 0.0),
        ];
        let src = vec![
            c(0.5, -1.0),
            c(2.0, 0.25),
            c(-0.5, 0.5),
            c(1.5, -0.25),
            c(0.75, 2.0),
        ];
        let base = vec![
            c(0.1, 0.2),
            c(0.2, 0.3),
            c(0.3, 0.4),
            c(0.4, 0.5),
            c(0.5, 0.6),
        ];
        let mut got = base.clone();
        phase_mul_accumulate_c64(&mut got, &phase, &src);
        for i in 0..got.len() {
            assert_c64_close(got[i], base[i] + phase[i] * src[i]);
        }
    }

    #[test]
    fn checked_phase_mul_accumulate_rejects_length_mismatch() {
        let mut dst = vec![c(1.0, 0.0); 2];
        let original = dst.clone();
        let error = try_phase_mul_accumulate_c64(&mut dst, &[c(1.0, 0.0)], &[c(1.0, 0.0); 2])
            .expect_err("phase update must reject mismatched lengths");
        assert!(matches!(
            error,
            crate::BackendError::InvalidArgument {
                operation: "phase_mul_accumulate_c64",
                ..
            }
        ));
        assert_eq!(dst, original);
    }

    #[test]
    fn legacy_phase_mul_accumulate_rejects_length_mismatch_without_mutation() {
        let mut dst = vec![c(1.0, 0.0); 2];
        let original = dst.clone();
        phase_mul_accumulate_c64(&mut dst, &[c(1.0, 0.0)], &[c(1.0, 0.0); 2]);
        assert_eq!(dst, original);
    }

    #[test]
    fn indexed_dense_matvec_gathers_source_values() {
        let matrix = vec![
            c(1.0, 0.0),
            c(0.0, 1.0),
            c(2.0, -1.0),
            c(-0.5, 0.25),
            c(0.75, 0.5),
            c(1.25, -0.75),
        ];
        let src = vec![c(10.0, 0.0), c(1.0, 2.0), c(-3.0, 0.5), c(0.25, -1.5)];
        let indices = vec![2, 0, 3];
        let mut got = vec![c(9.0, 9.0); 2];
        indexed_dense_matvec_c64(&matrix, 2, 3, 3, &src, &indices, &mut got);
        let x = [src[2], src[0], src[3]];
        assert_c64_close(
            got[0],
            matrix[0] * x[0] + matrix[1] * x[1] + matrix[2] * x[2],
        );
        assert_c64_close(
            got[1],
            matrix[3] * x[0] + matrix[4] * x[1] + matrix[5] * x[2],
        );
    }

    #[test]
    fn indexed_dense_matvec_accumulates_to_destination_indices() {
        let matrix = vec![c(1.0, 1.0), c(2.0, 0.0), c(-1.0, 0.5), c(0.25, -0.75)];
        let src = vec![c(2.0, -1.0), c(0.5, 3.0), c(-2.0, 0.25)];
        let src_indices = vec![2, 0];
        let dst_indices = vec![3, 1];
        let mut got = vec![c(0.0, 0.0); 5];
        got[1] = c(1.0, -1.0);
        indexed_dense_matvec_accumulate_c64(
            &matrix,
            2,
            2,
            2,
            &src,
            &src_indices,
            &mut got,
            &dst_indices,
        );
        let x = [src[2], src[0]];
        assert_c64_close(got[3], matrix[0] * x[0] + matrix[1] * x[1]);
        assert_c64_close(got[1], c(1.0, -1.0) + matrix[2] * x[0] + matrix[3] * x[1]);
    }

    #[test]
    fn gather_dot_scatter_batch_runs_multiple_blocks() {
        let src = vec![c(1.0, 0.0), c(0.0, 2.0), c(-1.0, 0.5)];
        let a0 = vec![c(2.0, 0.0), c(0.5, 0.5)];
        let a1 = vec![c(1.0, -1.0), c(-0.25, 0.75)];
        let batches = [
            GatherDotScatterC64 {
                matrix: &a0,
                rows: 1,
                cols: 2,
                lda: 2,
                src_indices: &[0, 1],
                dst_indices: &[2],
            },
            GatherDotScatterC64 {
                matrix: &a1,
                rows: 1,
                cols: 2,
                lda: 2,
                src_indices: &[2, 0],
                dst_indices: &[2],
            },
        ];
        let mut got = vec![c(0.0, 0.0); 3];
        gather_dot_scatter_batch_c64(&batches, &src, &mut got);
        let expected = a0[0] * src[0] + a0[1] * src[1] + a1[0] * src[2] + a1[1] * src[0];
        assert_c64_close(got[2], expected);
    }

    #[test]
    fn dense_matvec_scatter_uses_contiguous_source() {
        let matrix = vec![c(1.0, 0.0), c(0.0, 1.0), c(-0.5, 0.25), c(2.0, -1.0)];
        let src = vec![c(0.5, -1.0), c(3.0, 0.25)];
        let dst_indices = vec![4, 0];
        let mut got = vec![c(0.0, 0.0); 5];
        dense_matvec_scatter_accumulate_c64(&matrix, 2, 2, 2, &src, &mut got, &dst_indices);
        assert_c64_close(got[4], matrix[0] * src[0] + matrix[1] * src[1]);
        assert_c64_close(got[0], matrix[2] * src[0] + matrix[3] * src[1]);
    }

    #[test]
    fn compact_indexed_gather_selects_active_directions() {
        let src = vec![c(0.0, 0.0), c(1.0, -1.0), c(2.0, -2.0), c(3.0, -3.0)];
        let indices = vec![3, 1, 99];
        let mut got = vec![c(10.0, 10.0); 4];
        compact_indexed_gather_c64(&src, &indices, &mut got);
        assert_c64_close(got[0], src[3]);
        assert_c64_close(got[1], src[1]);
        assert_c64_close(got[2], c(10.0, 10.0));
        assert_c64_close(got[3], c(10.0, 10.0));
    }

    #[test]
    fn compact_indexed_scatter_accumulate_adds_active_directions() {
        let src = vec![c(1.0, 0.5), c(-2.0, 1.0), c(3.0, -1.5)];
        let indices = vec![2, 0, 99];
        let mut got = vec![c(0.25, -0.25); 4];
        compact_indexed_scatter_accumulate_c64(&src, &indices, &mut got);
        assert_c64_close(got[2], c(0.25, -0.25) + src[0]);
        assert_c64_close(got[0], c(0.25, -0.25) + src[1]);
        assert_c64_close(got[1], c(0.25, -0.25));
        assert_c64_close(got[3], c(0.25, -0.25));
    }
}
