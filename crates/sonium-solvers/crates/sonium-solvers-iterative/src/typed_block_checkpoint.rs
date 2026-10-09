//! Scalar-family-aware checkpoint codec for block FGMRES solutions.
//!
//! The legacy `BLKCP001` codec remains available for native `f64` callers.
//! This codec uses canonical `f64` real/imaginary pairs plus explicit scalar
//! metadata, so the same wire contract can safely carry `f32`, `f64`,
//! `Complex32`, and `Complex64` solutions without silently changing precision
//! on restore.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, ComplexField, SolverError},
};

use super::{BlockFgmresSolution, BlockGmresSolution, KrylovOpCounts};

/// Magic prefix for scalar-family-aware block-FGMRES packets.
pub const TYPED_BLOCK_FGMRES_CHECKPOINT_MAGIC: [u8; 8] = *b"BLKCP002";
/// Version for [`TYPED_BLOCK_FGMRES_CHECKPOINT_MAGIC`].
pub const TYPED_BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION: u16 = 1;

const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

impl<T: ComplexField> BlockFgmresSolution<T> {
    /// Serialize a block-FGMRES result with scalar-family metadata.
    ///
    /// Values are encoded as canonical `f64` real/imaginary pairs.  The
    /// scalar size, real-component size, and real/complex shape are recorded
    /// and must match the destination type on restore.
    pub fn to_typed_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate(self)?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&TYPED_BLOCK_FGMRES_CHECKPOINT_MAGIC);
        bytes.extend_from_slice(&TYPED_BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        bytes.push(
            u8::try_from(std::mem::size_of::<T>())
                .map_err(|_| checkpoint_error("scalar size does not fit checkpoint header"))?,
        );
        bytes
            .push(u8::try_from(std::mem::size_of::<T::Real>()).map_err(|_| {
                checkpoint_error("real scalar size does not fit checkpoint header")
            })?);
        bytes.push(scalar_is_complex::<T>());
        bytes.push(u8::from(self.converged));
        bytes.extend_from_slice(&[0, 0, 0]);
        push_u64(&mut bytes, self.iterations)?;

        push_u64(&mut bytes, self.x.nrows())?;
        push_u64(&mut bytes, self.x.ncols())?;
        for value in self.x.as_slice() {
            push_scalar(&mut bytes, *value)?;
        }

        push_real_vector(&mut bytes, &self.residuals)?;
        push_real_vector(&mut bytes, &self.true_residuals)?;
        push_real_vector(&mut bytes, &self.backward_errors)?;

        push_u64(&mut bytes, self.attainable_accuracy.len())?;
        for status in &self.attainable_accuracy {
            bytes.push(match status {
                AccuracyStatus::NotChecked => 0,
                AccuracyStatus::Attained => 1,
                AccuracyStatus::NotAttained => 2,
            });
        }

        for value in [
            self.op_counts.spmv_count,
            self.op_counts.preconditioner_count,
            self.op_counts.dot_reductions,
            self.op_counts.norm_count,
            self.op_counts.axpy_passes,
            self.op_counts.synchronization_points,
            self.op_counts.exact_residual_checks,
        ] {
            push_u64(&mut bytes, value)?;
        }
        bytes.extend_from_slice(&self.op_counts.basis_bytes_touched.to_le_bytes());
        Ok(bytes)
    }

    /// Restore a scalar-family-aware block-FGMRES result.
    pub fn from_typed_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != TYPED_BLOCK_FGMRES_CHECKPOINT_MAGIC {
            return Err(checkpoint_error(
                "invalid typed block-FGMRES checkpoint magic",
            ));
        }
        if reader.u16()? != TYPED_BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error(
                "unsupported typed block-FGMRES checkpoint version",
            ));
        }
        if reader.u8()? != u8::try_from(std::mem::size_of::<T>()).unwrap_or(u8::MAX)
            || reader.u8()? != u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX)
            || reader.u8()? != scalar_is_complex::<T>()
        {
            return Err(checkpoint_error("checkpoint scalar family does not match"));
        }
        let converged = match reader.u8()? {
            0 => false,
            1 => true,
            _ => return Err(checkpoint_error("invalid convergence flag")),
        };
        if reader.take(3)? != [0, 0, 0] {
            return Err(checkpoint_error("unsupported typed checkpoint flags"));
        }
        let iterations = reader.usize()?;

        let rows = reader.usize()?;
        let cols = reader.usize()?;
        let element_count = rows
            .checked_mul(cols)
            .ok_or_else(|| checkpoint_error("matrix dimensions overflow"))?;
        if element_count > MAX_ELEMENTS {
            return Err(checkpoint_error("matrix dimensions are unreasonable"));
        }
        let mut values = Vec::with_capacity(element_count);
        for _ in 0..element_count {
            values.push(reader.scalar::<T>()?);
        }
        let x = DenseMatrix::from_row_major_vec(rows, cols, values);

        let residuals = reader.real_vector::<T::Real>()?;
        let true_residuals = reader.real_vector::<T::Real>()?;
        let backward_errors = reader.real_vector::<T::Real>()?;

        let accuracy_len = reader.usize()?;
        if accuracy_len > MAX_ELEMENTS {
            return Err(checkpoint_error("accuracy metadata is unreasonable"));
        }
        let mut attainable_accuracy = Vec::with_capacity(accuracy_len);
        for _ in 0..accuracy_len {
            attainable_accuracy.push(match reader.u8()? {
                0 => AccuracyStatus::NotChecked,
                1 => AccuracyStatus::Attained,
                2 => AccuracyStatus::NotAttained,
                _ => return Err(checkpoint_error("invalid accuracy status")),
            });
        }

        let spmv_count = reader.usize()?;
        let preconditioner_count = reader.usize()?;
        let dot_reductions = reader.usize()?;
        let norm_count = reader.usize()?;
        let axpy_passes = reader.usize()?;
        let synchronization_points = reader.usize()?;
        let exact_residual_checks = reader.usize()?;
        let basis_bytes_touched = reader.u64()?;

        if !reader.is_empty() {
            return Err(checkpoint_error("trailing bytes"));
        }

        let result = Self {
            x,
            iterations,
            residuals,
            true_residuals,
            backward_errors,
            attainable_accuracy,
            converged,
            op_counts: KrylovOpCounts {
                spmv_count,
                preconditioner_count,
                dot_reductions,
                norm_count,
                axpy_passes,
                basis_bytes_touched,
                synchronization_points,
                exact_residual_checks,
            },
        };
        validate(&result)?;
        Ok(result)
    }
}

impl<T: ComplexField> BlockGmresSolution<T> {
    /// Serialize a block-GMRES result with the same scalar-family-aware wire
    /// contract as [`BlockFgmresSolution::to_typed_checkpoint_bytes`].
    pub fn to_typed_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        BlockFgmresSolution {
            x: self.x.clone(),
            iterations: self.iterations,
            residuals: self.residuals.clone(),
            true_residuals: self.true_residuals.clone(),
            backward_errors: self.backward_errors.clone(),
            attainable_accuracy: self.attainable_accuracy.clone(),
            converged: self.converged,
            op_counts: self.op_counts,
        }
        .to_typed_checkpoint_bytes()
    }

    /// Restore a block-GMRES result from the scalar-family-aware wire format.
    pub fn from_typed_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let decoded = BlockFgmresSolution::<T>::from_typed_checkpoint_bytes(bytes)?;
        Ok(Self {
            x: decoded.x,
            iterations: decoded.iterations,
            residuals: decoded.residuals,
            true_residuals: decoded.true_residuals,
            backward_errors: decoded.backward_errors,
            attainable_accuracy: decoded.attainable_accuracy,
            converged: decoded.converged,
            op_counts: decoded.op_counts,
        })
    }
}

fn validate<T: ComplexField>(result: &BlockFgmresSolution<T>) -> Result<(), SolverError> {
    let columns = result.x.ncols();
    if result.x.nrows().saturating_mul(columns) > MAX_ELEMENTS
        || result.residuals.len() != columns
        || result.true_residuals.len() != columns
        || result.backward_errors.len() != columns
        || result.attainable_accuracy.len() != columns
        || result
            .x
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        || result
            .residuals
            .iter()
            .chain(&result.true_residuals)
            .chain(&result.backward_errors)
            .any(|value| !value.is_finite() || *value < T::Real::zero())
    {
        return Err(checkpoint_error(
            "non-finite values or inconsistent result shape",
        ));
    }
    Ok(())
}

fn scalar_is_complex<T: ComplexField>() -> u8 {
    u8::from(T::from_re_im(T::Real::zero(), T::Real::one()).norm() > T::Real::zero())
}

fn push_u64(bytes: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    bytes.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| checkpoint_error("checkpoint value does not fit u64"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_scalar<T: ComplexField>(bytes: &mut Vec<u8>, value: T) -> Result<(), SolverError> {
    let re = value
        .re()
        .to_f64()
        .ok_or_else(|| checkpoint_error("real component cannot be represented as f64"))?;
    let im = value
        .im()
        .to_f64()
        .ok_or_else(|| checkpoint_error("imaginary component cannot be represented as f64"))?;
    if !re.is_finite() || !im.is_finite() {
        return Err(checkpoint_error("non-finite scalar component"));
    }
    bytes.extend_from_slice(&re.to_le_bytes());
    bytes.extend_from_slice(&im.to_le_bytes());
    Ok(())
}

fn push_real_vector<R: ToPrimitive + Copy + Float>(
    bytes: &mut Vec<u8>,
    values: &[R],
) -> Result<(), SolverError> {
    push_u64(bytes, values.len())?;
    for value in values {
        let value = value
            .to_f64()
            .ok_or_else(|| checkpoint_error("real component cannot be represented as f64"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(checkpoint_error("invalid residual diagnostic"));
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| checkpoint_error("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(checkpoint_error("truncated typed checkpoint"));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("checked length"),
        ))
    }

    fn u64(&mut self) -> Result<u64, SolverError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("checked length"),
        ))
    }

    fn usize(&mut self) -> Result<usize, SolverError> {
        usize::try_from(self.u64()?)
            .map_err(|_| checkpoint_error("checkpoint size does not fit usize"))
    }

    fn f64(&mut self) -> Result<f64, SolverError> {
        Ok(f64::from_le_bytes(
            self.take(8)?.try_into().expect("checked length"),
        ))
    }

    fn scalar<T: ComplexField>(&mut self) -> Result<T, SolverError> {
        let re = self.f64()?;
        let im = self.f64()?;
        if !re.is_finite() || !im.is_finite() {
            return Err(checkpoint_error("non-finite scalar component"));
        }
        if scalar_is_complex::<T>() == 0 && im != 0.0 {
            return Err(checkpoint_error(
                "real scalar has non-zero imaginary component",
            ));
        }
        let re = T::Real::from_f64(re)
            .ok_or_else(|| checkpoint_error("real component cannot be restored"))?;
        let im = T::Real::from_f64(im)
            .ok_or_else(|| checkpoint_error("imaginary component cannot be restored"))?;
        let value = T::from_re_im(re, im);
        if !value.norm().is_finite() {
            return Err(checkpoint_error("restored scalar is non-finite"));
        }
        Ok(value)
    }

    fn real_vector<R: Float + FromPrimitive>(&mut self) -> Result<Vec<R>, SolverError> {
        let length = self.usize()?;
        if length > MAX_ELEMENTS || length > self.bytes.len().saturating_sub(self.offset) / 8 {
            return Err(checkpoint_error("real-vector length is unreasonable"));
        }
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let value = self.f64()?;
            if !value.is_finite() || value < 0.0 {
                return Err(checkpoint_error("invalid residual diagnostic"));
            }
            values.push(
                R::from_f64(value)
                    .ok_or_else(|| checkpoint_error("real component cannot be restored"))?,
            );
        }
        Ok(values)
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn checkpoint_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "typed block FGMRES checkpoint",
        reason: reason.into(),
    }
}
