//! Deterministic segmented reductions used by element and boundary assembly.

use crate::DeviceScalar;
use thiserror::Error;

/// Validation failures for a segmented reduction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SegmentedReductionError {
    #[error("output length {output_len} overflows the offset count")]
    LengthOverflow { output_len: usize },
    #[error("offset array has length {actual}, expected {expected}")]
    OffsetCount { actual: usize, expected: usize },
    #[error("segment offset {offset} at position {position} is negative")]
    NegativeOffset { position: usize, offset: i32 },
    #[error("segment offsets are not monotone at position {position}: {previous} > {current}")]
    NonMonotone {
        position: usize,
        previous: usize,
        current: usize,
    },
    #[error("segment offset {offset} exceeds values length {values_len}")]
    OutOfBounds { offset: usize, values_len: usize },
}

/// Sum each half-open segment `values[offsets[i]..offsets[i + 1]]`.
///
/// The reduction is deterministic: values within each segment are consumed
/// in storage order, empty segments produce zero, and the output is fully
/// overwritten. `i32` offsets match the index buffers used by CUDA/Metal
/// sparse kernels.
/// Validate offset metadata without touching the values. Device backends use
/// this before launching a native kernel.
pub(crate) fn validate_segmented_offsets(
    values_len: usize,
    offsets: &[i32],
    output_len: usize,
) -> Result<(), SegmentedReductionError> {
    let expected_offsets = output_len
        .checked_add(1)
        .ok_or(SegmentedReductionError::LengthOverflow { output_len })?;
    if offsets.len() != expected_offsets {
        return Err(SegmentedReductionError::OffsetCount {
            actual: offsets.len(),
            expected: expected_offsets,
        });
    }

    let mut previous = 0usize;
    for (position, &raw_offset) in offsets.iter().enumerate() {
        if raw_offset < 0 {
            return Err(SegmentedReductionError::NegativeOffset {
                position,
                offset: raw_offset,
            });
        }
        let current = raw_offset as usize;
        if current < previous {
            return Err(SegmentedReductionError::NonMonotone {
                position,
                previous,
                current,
            });
        }
        if current > values_len {
            return Err(SegmentedReductionError::OutOfBounds {
                offset: current,
                values_len,
            });
        }
        previous = current;
    }
    Ok(())
}

pub fn segmented_sum<T: DeviceScalar>(
    values: &[T],
    offsets: &[i32],
    output: &mut [T],
) -> Result<(), SegmentedReductionError> {
    validate_segmented_offsets(values.len(), offsets, output.len())?;

    let expected_offsets =
        output
            .len()
            .checked_add(1)
            .ok_or(SegmentedReductionError::LengthOverflow {
                output_len: output.len(),
            })?;
    if offsets.len() != expected_offsets {
        return Err(SegmentedReductionError::OffsetCount {
            actual: offsets.len(),
            expected: expected_offsets,
        });
    }

    let mut previous = 0usize;
    for (position, &raw_offset) in offsets.iter().enumerate() {
        if raw_offset < 0 {
            return Err(SegmentedReductionError::NegativeOffset {
                position,
                offset: raw_offset,
            });
        }
        let current = raw_offset as usize;
        if current < previous {
            return Err(SegmentedReductionError::NonMonotone {
                position,
                previous,
                current,
            });
        }
        if current > values.len() {
            return Err(SegmentedReductionError::OutOfBounds {
                offset: current,
                values_len: values.len(),
            });
        }
        previous = current;
    }

    for (segment, result) in output.iter_mut().enumerate() {
        let start = offsets[segment] as usize;
        let end = offsets[segment + 1] as usize;
        let mut sum = T::zero();
        for &value in &values[start..end] {
            sum = sum + value;
        }
        *result = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn sums_empty_and_nonempty_segments_in_order() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let mut output = [99.0; 4];
        segmented_sum(&values, &[0, 2, 2, 3, 4], &mut output).expect("valid segments");
        assert_eq!(output, [3.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn supports_complex_values() {
        let values = [Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
        let mut output = [Complex64::default()];
        segmented_sum(&values, &[0, 2], &mut output).expect("valid complex segment");
        assert_eq!(output[0], Complex64::new(4.0, 1.0));
    }

    #[test]
    fn rejects_malformed_offsets() {
        let mut output = [0.0; 2];
        assert!(matches!(
            segmented_sum(&[1.0], &[0, 2, 1], &mut output),
            Err(SegmentedReductionError::OutOfBounds { .. })
        ));
        assert!(matches!(
            segmented_sum(&[1.0], &[0, -1, 1], &mut output),
            Err(SegmentedReductionError::NegativeOffset { .. })
        ));
        assert!(matches!(
            segmented_sum(&[1.0], &[0, 1], &mut output),
            Err(SegmentedReductionError::OffsetCount { .. })
        ));
    }
}
