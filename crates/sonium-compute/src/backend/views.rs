//! Dense-view descriptors and buffer-view handles.
//!
//! Moved verbatim from `backend/mod.rs`; the checked entry points
//! backing these descriptors live on [`super::Backend`].

use crate::error::{BackendError, BackendResult};
use crate::scalar::BufferElement;

/// Operation applied to a dense matrix in BLAS-2/3 calls.
///
/// Mirrors the classical BLAS `op(A)` parameter:
/// - [`NoTrans`](Self::NoTrans): `op(A) = A`
/// - [`Trans`](Self::Trans): `op(A) = A^T`
/// - [`ConjTrans`](Self::ConjTrans): `op(A) = A^H` (transpose then conjugate)
///
/// For real scalar types, `Trans` and `ConjTrans` are equivalent; backends
/// must accept either and behave identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transpose {
    NoTrans,
    Trans,
    ConjTrans,
}

/// Aliasing contract attached to a checked view descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewAliasing {
    /// The operation only reads this view, so it may overlap another reader.
    ReadOnly,
    /// The input and output storage must not overlap. This is a contract
    /// marker; a `ViewDescriptor` does not carry allocation identity and
    /// therefore cannot prove disjointness by itself. Checked public APIs
    /// establish this through their Rust buffer borrowing contract.
    Disjoint,
    /// Overlap is supported by the operation's defined staging semantics.
    OverlapSafe,
}

/// Logical memory layout carried by a rank-N view descriptor.
///
/// `Strided` is the conservative form used when a view has arbitrary
/// (including padded) element strides.  The row/column-major variants are
/// stronger contracts for native tensor bindings: [`ViewDescriptor::validate`]
/// verifies that their strides agree with the declared shape before a backend
/// can use the descriptor to launch a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewLayout {
    Strided,
    RowMajor,
    ColumnMajor,
}

/// Backend-neutral binding descriptor for a one-dimensional view.
///
/// `BufferView` is the compact element-index form used by callers. This
/// descriptor is the form that a native binding can validate and translate to
/// a pointer: rank, shape, byte offset, element strides, scalar type, device,
/// and aliasing policy are all explicit. `aliasing` describes the operation
/// contract; it is deliberately not an allocation identity. Backend buffer
/// handles have private storage, so safe callers cannot construct two views
/// into one allocation except through an API that explicitly provides
/// overlap-safe semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewDescriptor {
    pub rank: usize,
    pub shape: Vec<usize>,
    pub byte_offset: usize,
    pub element_strides: Vec<usize>,
    pub layout: ViewLayout,
    pub dtype: &'static str,
    pub device: &'static str,
    pub aliasing: ViewAliasing,
}

impl ViewDescriptor {
    /// Build and validate a descriptor while preserving the checked API's
    /// allocation-failure contract. Public callers may still construct the
    /// struct directly when they already own the vectors; backend boundary
    /// helpers should prefer this constructor for borrowed shape metadata.
    pub fn try_new(
        shape: &[usize],
        byte_offset: usize,
        element_strides: &[usize],
        layout: ViewLayout,
        dtype: &'static str,
        device: &'static str,
        aliasing: ViewAliasing,
    ) -> BackendResult<Self> {
        let mut owned_shape = Vec::new();
        owned_shape
            .try_reserve_exact(shape.len())
            .map_err(|_| BackendError::Allocation {
                reason: "view_descriptor: shape allocation failed".into(),
            })?;
        owned_shape.extend_from_slice(shape);

        let mut owned_strides = Vec::new();
        owned_strides
            .try_reserve_exact(element_strides.len())
            .map_err(|_| BackendError::Allocation {
                reason: "view_descriptor: stride allocation failed".into(),
            })?;
        owned_strides.extend_from_slice(element_strides);

        let descriptor = Self {
            rank: owned_shape.len(),
            shape: owned_shape,
            byte_offset,
            element_strides: owned_strides,
            layout,
            dtype,
            device,
            aliasing,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Validate descriptor metadata before it is handed to a native binding.
    ///
    /// The descriptor is intentionally public so backend adapters can inspect
    /// it, but public fields must not turn an invalid rank/shape/stride tuple
    /// into an unchecked kernel launch. Bounds against a particular allocation
    /// are validated by [`super::Backend::checked_view_descriptor`].
    pub fn validate(&self) -> BackendResult<()> {
        if self.rank == 0 {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "rank must be non-zero".into(),
            });
        }
        if self.shape.len() != self.rank {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "shape rank {} does not match descriptor rank {}",
                    self.shape.len(),
                    self.rank
                ),
            });
        }
        if self.element_strides.len() != self.rank {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "stride rank {} does not match descriptor rank {}",
                    self.element_strides.len(),
                    self.rank
                ),
            });
        }
        if self.dtype.is_empty() {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "dtype must not be empty".into(),
            });
        }
        if self.device.is_empty() {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "device must not be empty".into(),
            });
        }
        if self.element_strides.contains(&0) {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "element strides must be non-zero".into(),
            });
        }
        if !self
            .layout
            .matches_shape_and_strides(&self.shape, &self.element_strides)
        {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "declared {:?} layout does not match shape/element strides",
                    self.layout
                ),
            });
        }
        Ok(())
    }

    /// Check the explicit layout contract without performing byte arithmetic.
    /// Return the number of logical elements selected by this descriptor.
    pub fn checked_element_count(&self) -> BackendResult<usize> {
        self.validate()?;
        self.shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "view_descriptor",
                    reason: "logical element count overflows usize".into(),
                })
        })
    }

    /// Map a row-major logical index to the selected buffer element.
    ///
    /// The descriptor's strides describe physical storage; logical iteration
    /// remains row-major over `shape`.  This is the common staging order used
    /// by checked N-D copies on every backend.
    pub fn checked_element_index<T: BufferElement>(
        &self,
        logical_index: usize,
        device: &'static str,
    ) -> BackendResult<usize> {
        self.validate_for::<T>(device)?;
        let count = self.checked_element_count()?;
        if logical_index >= count {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "logical element index {logical_index} is outside descriptor length {count}"
                ),
            });
        }

        let mut remainder = logical_index;
        let mut element_offset = self.byte_offset / std::mem::size_of::<T>();
        for axis in (0..self.rank).rev() {
            let coordinate = remainder % self.shape[axis];
            remainder /= self.shape[axis];
            let axis_offset = coordinate
                .checked_mul(self.element_strides[axis])
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "view_descriptor",
                    reason: "logical coordinate stride overflows buffer indexing".into(),
                })?;
            element_offset = element_offset.checked_add(axis_offset).ok_or_else(|| {
                BackendError::InvalidArgument {
                    operation: "view_descriptor",
                    reason: "logical coordinate offset overflows buffer indexing".into(),
                }
            })?;
        }
        Ok(element_offset)
    }

    /// Validate a descriptor against its scalar type and backend binding.
    pub fn validate_for<T: BufferElement>(&self, device: &'static str) -> BackendResult<()> {
        self.validate()?;
        let expected_dtype = std::any::type_name::<T>();
        if self.dtype != expected_dtype {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "dtype {} does not match buffer element type {}",
                    self.dtype, expected_dtype
                ),
            });
        }
        if self.device != device {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "device {} does not match backend device {}",
                    self.device, device
                ),
            });
        }
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "zero-sized buffer elements are not bindable".into(),
            });
        }
        if !self.byte_offset.is_multiple_of(element_size) {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "byte offset {} is not aligned to element size {}",
                    self.byte_offset, element_size
                ),
            });
        }
        self.byte_offset
            .checked_add(element_size.saturating_sub(1))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "byte offset overflows address arithmetic".into(),
            })?;
        for stride in &self.element_strides {
            stride
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "view_descriptor",
                    reason: "byte stride overflows address arithmetic".into(),
                })?;
        }
        self.checked_byte_end_exclusive_unchecked::<T>()?;
        Ok(())
    }

    /// Return the exclusive byte bound touched by this descriptor.
    ///
    /// Empty shapes touch no storage and therefore return `byte_offset`.
    pub fn checked_byte_end_exclusive<T: BufferElement>(
        &self,
        device: &'static str,
    ) -> BackendResult<usize> {
        self.validate_for::<T>(device)?;
        self.checked_byte_end_exclusive_unchecked::<T>()
    }

    /// Validate the descriptor against a concrete allocation.
    ///
    /// Metadata validation alone cannot prove that a native binding stays
    /// inside its buffer.  Keep that second check at the shared boundary so
    /// multidimensional descriptors cannot silently bypass the allocation
    /// bound that [`BufferView::new`] provides for one-dimensional views.
    pub fn validate_for_buffer<T: BufferElement>(
        &self,
        buffer_len: usize,
        device: &'static str,
    ) -> BackendResult<()> {
        self.validate_for::<T>(device)?;
        let allocation_bytes = buffer_len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "buffer byte length overflows address arithmetic".into(),
            })?;
        let byte_end = self.checked_byte_end_exclusive_unchecked::<T>()?;
        if byte_end > allocation_bytes {
            return Err(BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: format!(
                    "descriptor byte range ends at {byte_end}, beyond buffer allocation of {allocation_bytes} bytes"
                ),
            });
        }
        Ok(())
    }

    fn checked_byte_end_exclusive_unchecked<T: BufferElement>(&self) -> BackendResult<usize> {
        let element_size = std::mem::size_of::<T>();
        if self.shape.contains(&0) {
            return Ok(self.byte_offset);
        }
        let max_element_offset = self
            .shape
            .iter()
            .zip(self.element_strides.iter())
            .try_fold(0usize, |offset, (shape, stride)| {
                shape
                    .checked_sub(1)
                    .and_then(|last| last.checked_mul(*stride))
                    .and_then(|axis_offset| offset.checked_add(axis_offset))
            })
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "shape and element strides overflow address arithmetic".into(),
            })?;
        self.byte_offset
            .checked_add(
                max_element_offset
                    .checked_mul(element_size)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation: "view_descriptor",
                        reason: "shape byte span overflows address arithmetic".into(),
                    })?,
            )
            .and_then(|last| last.checked_add(element_size))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "view_descriptor",
                reason: "descriptor byte end overflows address arithmetic".into(),
            })
    }
}

impl ViewLayout {
    fn matches_shape_and_strides(self, shape: &[usize], strides: &[usize]) -> bool {
        if matches!(self, Self::Strided) {
            return true;
        }
        if shape.len() != strides.len() {
            return false;
        }
        // An empty extent has no addressable element, so there is no unique
        // contiguous stride convention to enforce.  Keep the metadata valid
        // while still requiring the rank/stride vectors to agree above.
        if shape.contains(&0) {
            return true;
        }
        match self {
            Self::Strided => true,
            Self::RowMajor => {
                let mut expected = 1usize;
                for axis in (0..shape.len()).rev() {
                    if strides[axis] != expected {
                        return false;
                    }
                    expected = match expected.checked_mul(shape[axis]) {
                        Some(value) => value,
                        None => return false,
                    };
                }
                true
            }
            Self::ColumnMajor => {
                let mut expected = 1usize;
                for axis in 0..shape.len() {
                    if strides[axis] != expected {
                        return false;
                    }
                    expected = match expected.checked_mul(shape[axis]) {
                        Some(value) => value,
                        None => return false,
                    };
                }
                true
            }
        }
    }
}

/// Build the checked rank-2 descriptor used by native batched FFT kernels.
///
/// FFT backends operate on a contiguous row-major `[batch, fft_len]` view and
/// update that view in place.  Keeping this admission boundary shared prevents
/// CUDA/cuFFT and Metal/Stockham from drifting on rank, stride, device, or
/// aliasing validation before they hand raw pointers to native code.
pub fn checked_fft_view_descriptor<T: BufferElement>(
    buffer_len: usize,
    fft_len: usize,
    device: &'static str,
) -> BackendResult<ViewDescriptor> {
    if fft_len == 0 {
        return Err(BackendError::InvalidArgument {
            operation: "fft_view_descriptor",
            reason: "FFT length must be greater than zero".into(),
        });
    }
    if !buffer_len.is_multiple_of(fft_len) {
        return Err(BackendError::InvalidArgument {
            operation: "fft_view_descriptor",
            reason: format!("buffer length {buffer_len} is not divisible by FFT length {fft_len}"),
        });
    }
    let batch = buffer_len / fft_len;
    let descriptor = ViewDescriptor {
        rank: 2,
        shape: vec![batch, fft_len],
        byte_offset: 0,
        element_strides: vec![fft_len, 1],
        layout: ViewLayout::RowMajor,
        dtype: std::any::type_name::<T>(),
        device,
        aliasing: ViewAliasing::OverlapSafe,
    };
    descriptor.validate_for_buffer::<T>(buffer_len, device)?;
    Ok(descriptor)
}

/// Build the checked rank-2 descriptor used by native strided batched FFTs.
///
/// Each transform is contiguous, while consecutive transforms begin `stride`
/// elements apart. The gaps between transforms are not touched.
pub fn checked_fft_strided_view_descriptor<T: BufferElement>(
    buffer_len: usize,
    fft_len: usize,
    batch: usize,
    stride: usize,
    device: &'static str,
) -> BackendResult<ViewDescriptor> {
    if fft_len == 0 || stride < fft_len {
        return Err(BackendError::InvalidArgument {
            operation: "fft_strided_view_descriptor",
            reason: "FFT length must be positive and stride must cover one transform".into(),
        });
    }
    let required = if batch == 0 {
        0
    } else {
        batch
            .checked_sub(1)
            .and_then(|last| last.checked_mul(stride))
            .and_then(|last| last.checked_add(fft_len))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "fft_strided_view_descriptor",
                reason: "FFT batch shape overflows address arithmetic".into(),
            })?
    };
    if buffer_len < required {
        return Err(BackendError::InvalidArgument {
            operation: "fft_strided_view_descriptor",
            reason: format!(
                "buffer length {buffer_len} is shorter than required strided span {required}"
            ),
        });
    }
    let descriptor = ViewDescriptor {
        rank: 2,
        shape: vec![batch, fft_len],
        byte_offset: 0,
        element_strides: vec![stride, 1],
        layout: ViewLayout::Strided,
        dtype: std::any::type_name::<T>(),
        device,
        aliasing: ViewAliasing::OverlapSafe,
    };
    descriptor.validate_for_buffer::<T>(buffer_len, device)?;
    Ok(descriptor)
}

/// A checked one-dimensional view into a backend buffer.
///
/// Offsets and strides are measured in elements. `len == 0` is a valid empty
/// view at any offset in `[0, buffer_len]`; it touches no element regardless
/// of its stride. Native kernels must bind the corresponding
/// [`ViewDescriptor`] before constructing a raw pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferView {
    pub offset: usize,
    pub len: usize,
    pub stride: usize,
}

impl BufferView {
    fn validate_metadata(&self) -> BackendResult<()> {
        if self.stride == 0 {
            return Err(BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: "stride must be non-zero".into(),
            });
        }
        Ok(())
    }

    /// Validate a view against a buffer containing `buffer_len` elements.
    pub fn new(buffer_len: usize, offset: usize, len: usize, stride: usize) -> BackendResult<Self> {
        if stride == 0 {
            return Err(BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: "stride must be non-zero".into(),
            });
        }
        if offset > buffer_len {
            return Err(BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: format!("offset {offset} exceeds buffer length {buffer_len}"),
            });
        }
        let view = Self {
            offset,
            len,
            stride,
        };
        if len == 0 {
            return Ok(view);
        }
        if view.checked_end_exclusive()? > buffer_len {
            return Err(BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: format!(
                    "view [{offset}, len={len}, stride={stride}] exceeds buffer length {buffer_len}"
                ),
            });
        }
        Ok(view)
    }

    pub const fn contiguous(offset: usize, len: usize) -> Self {
        Self {
            offset,
            len,
            stride: 1,
        }
    }

    /// Return the exclusive element bound touched by this view.
    pub fn checked_end_exclusive(&self) -> BackendResult<usize> {
        self.validate_metadata()?;
        if self.len == 0 {
            return Ok(self.offset);
        }
        self.len
            .checked_sub(1)
            .and_then(|last| last.checked_mul(self.stride))
            .and_then(|last| self.offset.checked_add(last))
            .and_then(|last| last.checked_add(1))
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: format!(
                    "view [{}, len={}, stride={}] overflows buffer indexing",
                    self.offset, self.len, self.stride
                ),
            })
    }

    /// Return the element selected by `index`, rejecting invalid public
    /// struct literals as well as checked views.
    pub fn checked_element_index(&self, index: usize) -> BackendResult<usize> {
        self.validate_metadata()?;
        if index >= self.len {
            return Err(BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: format!("element index {index} is outside view length {}", self.len),
            });
        }
        let displacement =
            index
                .checked_mul(self.stride)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "buffer_view",
                    reason: "view element index multiplication overflow".into(),
                })?;
        self.offset
            .checked_add(displacement)
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "buffer_view",
                reason: "view element index addition overflow".into(),
            })
    }

    /// Return a native-binding descriptor after checking byte arithmetic.
    pub fn descriptor<T: BufferElement>(
        &self,
        device: &'static str,
    ) -> BackendResult<ViewDescriptor> {
        self.descriptor_with_aliasing::<T>(device, ViewAliasing::ReadOnly)
    }

    /// Return a native-binding descriptor with an explicit aliasing policy.
    pub fn descriptor_with_aliasing<T: BufferElement>(
        &self,
        device: &'static str,
        aliasing: ViewAliasing,
    ) -> BackendResult<ViewDescriptor> {
        self.validate_metadata()?;
        let element_size = std::mem::size_of::<T>();
        let byte_offset =
            self.offset
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "buffer_view",
                    reason: "view byte offset overflow".into(),
                })?;
        // Native bindings multiply the element stride by the scalar size when
        // forming an address. Check that conversion at the common boundary,
        // even though the shared descriptor keeps the stride in elements.
        let _byte_stride =
            self.stride
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation: "buffer_view",
                    reason: "view byte stride overflow".into(),
                })?;
        let descriptor = ViewDescriptor {
            rank: 1,
            shape: vec![self.len],
            byte_offset,
            element_strides: vec![self.stride],
            layout: ViewLayout::Strided,
            dtype: std::any::type_name::<T>(),
            device,
            aliasing,
        };
        descriptor.validate_for::<T>(device)?;
        Ok(descriptor)
    }

    /// Determine exactly whether two views into the same allocation select
    /// at least one common element.
    ///
    /// A view selects a bounded arithmetic progression, so comparing only
    /// the address intervals is insufficient: even and odd strided views
    /// can have overlapping intervals without sharing an element. The
    /// congruence check below keeps the result exact without enumerating
    /// either view.
    pub fn may_overlap(&self, other: &Self) -> BackendResult<bool> {
        let self_end = self.checked_end_exclusive()?;
        let other_end = other.checked_end_exclusive()?;
        if self.len == 0 || other.len == 0 {
            return Ok(false);
        }

        let lower = self.offset.max(other.offset) as u128;
        let upper = self_end.min(other_end).saturating_sub(1) as u128;
        if lower > upper {
            return Ok(false);
        }

        let first_offset = self.offset as u128;
        let second_offset = other.offset as u128;
        let first_stride = self.stride as u128;
        let second_stride = other.stride as u128;
        let gcd = Self::gcd(first_stride, second_stride);

        // A common element must satisfy both congruences. The residues are
        // compatible exactly when their offset difference is divisible by
        // the strides' gcd.
        if !first_offset.abs_diff(second_offset).is_multiple_of(gcd) {
            return Ok(false);
        }

        // Write a common element as `first_offset + first_stride * k` and
        // solve for k modulo `second_stride / gcd`.
        let reduced_second_stride = second_stride / gcd;
        let lower_delta = lower - first_offset;
        let first_k =
            lower_delta / first_stride + u128::from(!lower_delta.is_multiple_of(first_stride));
        let residue = if reduced_second_stride == 1 {
            0
        } else {
            let difference = if second_offset >= first_offset {
                ((second_offset - first_offset) / gcd) % reduced_second_stride
            } else {
                let magnitude = ((first_offset - second_offset) / gcd) % reduced_second_stride;
                if magnitude == 0 {
                    0
                } else {
                    reduced_second_stride - magnitude
                }
            };
            let Some(inverse) = Self::modular_inverse(
                (first_stride / gcd) % reduced_second_stride,
                reduced_second_stride,
            ) else {
                return Err(BackendError::InvalidArgument {
                    operation: "buffer_view",
                    reason: "failed to solve view-overlap congruence".into(),
                });
            };
            (difference * inverse) % reduced_second_stride
        };
        let step = reduced_second_stride;
        let k = if first_k <= residue {
            residue
        } else {
            let distance = first_k - residue;
            residue + (distance / step + u128::from(!distance.is_multiple_of(step))) * step
        };
        let Some(stride_offset) = first_stride.checked_mul(k) else {
            return Ok(false);
        };
        let Some(common) = first_offset.checked_add(stride_offset) else {
            return Ok(false);
        };

        Ok(common <= upper)
    }

    fn gcd(mut left: u128, mut right: u128) -> u128 {
        while right != 0 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        left
    }

    fn modular_inverse(value: u128, modulus: u128) -> Option<u128> {
        debug_assert!(modulus > 1);

        // All BufferView metadata originates from usize, which is at most
        // 64 bits on supported Rust targets. i128 therefore safely holds
        // the signed extended-Euclidean coefficients.
        let mut old_remainder = value as i128;
        let mut remainder = modulus as i128;
        let mut old_coefficient = 1i128;
        let mut coefficient = 0i128;

        while remainder != 0 {
            let quotient = old_remainder / remainder;
            (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
            (old_coefficient, coefficient) =
                (coefficient, old_coefficient - quotient * coefficient);
        }

        (old_remainder == 1).then(|| old_coefficient.rem_euclid(modulus as i128) as u128)
    }
}
