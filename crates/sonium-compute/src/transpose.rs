//! Cache-blocked row-major matrix transposition.
//!
//! The primitive is deliberately backend-neutral: callers can use it for
//! host/shared-memory staging when a backend does not expose a native
//! transpose kernel, while preserving one checked shape contract for CPU and
//! accelerator implementations.

use std::fmt;

/// Errors returned by [`transpose_2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeError {
    /// `rows * cols` cannot be represented by `usize`.
    LengthOverflow,
    /// The source buffer is shorter than the requested matrix.
    InputTooShort { required: usize, actual: usize },
    /// The destination buffer is shorter than the transposed matrix.
    OutputTooShort { required: usize, actual: usize },
    /// The source and destination ranges overlap.
    OverlappingBuffers,
}

impl fmt::Display for TransposeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthOverflow => formatter.write_str("transpose matrix length overflows usize"),
            Self::InputTooShort { required, actual } => {
                write!(
                    formatter,
                    "transpose input has {actual} elements, needs {required}"
                )
            }
            Self::OutputTooShort { required, actual } => {
                write!(
                    formatter,
                    "transpose output has {actual} elements, needs {required}"
                )
            }
            Self::OverlappingBuffers => {
                formatter.write_str("transpose input and output buffers overlap")
            }
        }
    }
}

impl std::error::Error for TransposeError {}

/// Transpose a row-major `rows × cols` matrix into a row-major `cols × rows`
/// matrix.
///
/// Only the first `rows * cols` elements of either buffer are accessed; extra
/// capacity is permitted. The buffers must not overlap. A tiled traversal
/// keeps both source and destination accesses cache-friendly for the medium
/// matrices used by FFT six-step and multidimensional transforms.
pub fn transpose_2d<T: Copy>(
    input: &[T],
    output: &mut [T],
    rows: usize,
    cols: usize,
) -> Result<(), TransposeError> {
    // A zero-row or zero-column matrix has no elements and therefore needs no
    // buffer access. Treat it as a valid no-op, matching the empty-shape
    // convention used by the checked BLAS operations.
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let required = rows
        .checked_mul(cols)
        .ok_or(TransposeError::LengthOverflow)?;
    if input.len() < required {
        return Err(TransposeError::InputTooShort {
            required,
            actual: input.len(),
        });
    }
    if output.len() < required {
        return Err(TransposeError::OutputTooShort {
            required,
            actual: output.len(),
        });
    }

    // The tiled kernel reads from `input` while writing `output`; even a
    // partially overlapping range would make later tiles observe overwritten
    // source values. Reject the aliasing contract before the first access.
    let bytes = required
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(TransposeError::LengthOverflow)?;
    if bytes != 0 {
        let input_start = input.as_ptr() as usize;
        let output_start = output.as_mut_ptr() as usize;
        let input_end = input_start
            .checked_add(bytes)
            .ok_or(TransposeError::LengthOverflow)?;
        let output_end = output_start
            .checked_add(bytes)
            .ok_or(TransposeError::LengthOverflow)?;
        if input_start < output_end && output_start < input_end {
            return Err(TransposeError::OverlappingBuffers);
        }
    }

    // 32 elements is small enough for the common f32/f64/Complex cache lines
    // while reducing loop overhead for the tiny matrices used in FFT plans.
    const TILE: usize = 32;
    let mut row0 = 0;
    while row0 < rows {
        let row_end = (row0 + TILE).min(rows);
        let mut col0 = 0;
        while col0 < cols {
            let col_end = (col0 + TILE).min(cols);
            for row in row0..row_end {
                let source_row = row * cols;
                for col in col0..col_end {
                    output[col * rows + row] = input[source_row + col];
                }
            }
            col0 += TILE;
        }
        row0 += TILE;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposes_rectangular_and_tiled_shapes() {
        for &(rows, cols) in &[(1, 1), (2, 3), (3, 2), (7, 33), (35, 65)] {
            let input: Vec<usize> = (0..rows * cols).collect();
            let mut output = vec![usize::MAX; rows * cols + 5];
            transpose_2d(&input, &mut output, rows, cols).unwrap();
            for row in 0..rows {
                for col in 0..cols {
                    assert_eq!(output[col * rows + row], input[row * cols + col]);
                }
            }
            assert_eq!(&output[rows * cols..], &[usize::MAX; 5]);
        }
    }

    #[test]
    fn rejects_invalid_shapes_and_short_buffers() {
        let input = [1_u32, 2, 3, 4];
        let mut output = [0_u32; 4];
        assert_eq!(transpose_2d(&input, &mut output, 0, 4), Ok(()));
        assert_eq!(
            transpose_2d(&input[..3], &mut output, 2, 2),
            Err(TransposeError::InputTooShort {
                required: 4,
                actual: 3
            })
        );
        assert_eq!(
            transpose_2d(&input, &mut output[..3], 2, 2),
            Err(TransposeError::OutputTooShort {
                required: 4,
                actual: 3
            })
        );
        assert_eq!(
            transpose_2d(&input, &mut output, usize::MAX, 2),
            Err(TransposeError::LengthOverflow)
        );
    }

    #[test]
    fn rejects_overlapping_source_and_destination_before_writing() {
        let mut storage = [1_u32, 2, 3, 4];
        let pointer = storage.as_mut_ptr();
        // SAFETY: the checked function must inspect the ranges and return
        // before dereferencing either alias. The test deliberately models a
        // caller that violated the non-overlap precondition.
        let input = unsafe { std::slice::from_raw_parts(pointer.cast_const(), 4) };
        // SAFETY: see the safety argument above; no element is read or
        // written after the overlap is detected.
        let output = unsafe { std::slice::from_raw_parts_mut(pointer, 4) };
        assert_eq!(
            transpose_2d(input, output, 2, 2),
            Err(TransposeError::OverlappingBuffers)
        );
        assert_eq!(storage, [1, 2, 3, 4]);
    }
}
