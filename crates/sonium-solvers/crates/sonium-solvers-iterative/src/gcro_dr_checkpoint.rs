//! Portable checkpoints for GCRO-DR recycled subspaces.
//!
//! A GCRO-DR subspace is reusable spectral state rather than a solution
//! vector.  This module keeps that state explicit: checkpoints validate
//! dimensions and finiteness, use a versioned little-endian wire format, and
//! restore into a fresh subspace whose companion `C = M⁻¹ A U` is rebuilt by
//! the next solve for the current operator and preconditioner.

use super::gcro_dr::{GcroDrConfig, GcroDrSubspace, try_gcro_dr, try_gcro_dr_preconditioned};
use super::gmres::GmresSolution;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_solvers_core::traits::{ComplexField, LinearOperator, Preconditioner, SolverError};

/// Version of the portable GCRO-DR checkpoint format.
pub const GCRO_DR_CHECKPOINT_FORMAT_VERSION: u16 = 1;

const CHECKPOINT_MAGIC: &[u8; 8] = b"SGCROCP1";

/// A validated GCRO-DR recycled-subspace checkpoint.
#[derive(Debug, Clone)]
pub struct GcroDrCheckpoint<T: ComplexField> {
    dimension: usize,
    target_size: usize,
    vectors: Vec<Vec<T>>,
}

impl<T: ComplexField> GcroDrCheckpoint<T> {
    /// Capture the current recycled subspace.
    pub fn from_subspace(subspace: &GcroDrSubspace<T>) -> Result<Self, SolverError> {
        Self::from_vectors(subspace.target_size(), subspace.vectors())
    }

    /// Build a checkpoint from explicit recycled vectors.
    pub fn from_vectors(target_size: usize, vectors: &[Vec<T>]) -> Result<Self, SolverError> {
        if vectors.len() > target_size {
            return Err(checkpoint_error(
                "recycled vector count exceeds configured target size",
            ));
        }
        let dimension = vectors.first().map_or(0, Vec::len);
        if vectors.iter().any(|vector| {
            vector.len() != dimension || vector.iter().any(|value| !value.norm().is_finite())
        }) {
            return Err(checkpoint_error(
                "recycled vectors have inconsistent dimensions or non-finite values",
            ));
        }
        Ok(Self {
            dimension,
            target_size,
            vectors: vectors.to_vec(),
        })
    }

    /// Number of entries in each stored recycled vector.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Configured number of vectors retained after each solve.
    pub const fn target_size(&self) -> usize {
        self.target_size
    }

    /// Number of vectors currently stored.
    pub const fn num_vectors(&self) -> usize {
        self.vectors.len()
    }

    /// Number of bytes occupied by vector values in memory.
    pub fn storage_bytes(&self) -> usize {
        self.vectors
            .iter()
            .map(|vector| vector.len().saturating_mul(std::mem::size_of::<T>()))
            .fold(0, usize::saturating_add)
    }

    /// Borrow the validated recycled vectors.
    pub fn vectors(&self) -> &[Vec<T>] {
        &self.vectors
    }

    /// Restore this checkpoint into a mutable subspace.
    pub fn restore_into(&self, subspace: &mut GcroDrSubspace<T>) -> Result<(), SolverError> {
        let restored = Self::from_vectors(self.target_size, &self.vectors)?;
        subspace.replace_vectors(restored.target_size, restored.vectors);
        Ok(())
    }

    /// Create a fresh subspace from this checkpoint.
    pub fn into_subspace(&self) -> GcroDrSubspace<T> {
        let mut subspace = GcroDrSubspace::new(self.target_size);
        // The checkpoint was validated at construction or decoding time.
        subspace.replace_vectors(self.target_size, self.vectors.clone());
        subspace
    }

    /// Resume an identity-preconditioned GCRO-DR solve and return its updated
    /// recycled state together with the checked result.
    pub fn resume<A>(
        &self,
        operator: &A,
        rhs: &[T],
        initial_guess: Option<&[T]>,
        config: &GcroDrConfig<T::Real>,
    ) -> Result<(GcroDrSubspace<T>, GmresSolution<T>), SolverError>
    where
        A: LinearOperator<T>,
    {
        let mut subspace = self.into_subspace();
        let result = try_gcro_dr(operator, rhs, initial_guess, &mut subspace, config)?;
        Ok((subspace, result))
    }

    /// Resume a left-preconditioned GCRO-DR solve and return updated state.
    pub fn resume_preconditioned<A, P>(
        &self,
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        initial_guess: Option<&[T]>,
        config: &GcroDrConfig<T::Real>,
    ) -> Result<(GcroDrSubspace<T>, GmresSolution<T>), SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        let mut subspace = self.into_subspace();
        let result = try_gcro_dr_preconditioned(
            operator,
            preconditioner,
            rhs,
            initial_guess,
            &mut subspace,
            config,
        )?;
        Ok((subspace, result))
    }

    /// Serialize the checkpoint using a versioned, little-endian format.
    /// Numeric entries are canonical `f64` real/imaginary pairs so a
    /// checkpoint can be exchanged between f32/f64 builds only when the
    /// scalar-size and real/complex markers match on restore.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let validated = Self::from_vectors(self.target_size, &self.vectors)?;
        let mut output = Vec::new();
        output.extend_from_slice(CHECKPOINT_MAGIC);
        output.extend_from_slice(&GCRO_DR_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(std::mem::size_of::<T>() as u8);
        output.push(std::mem::size_of::<T::Real>() as u8);
        output.push(scalar_is_complex::<T>());
        output.push(0);
        push_u64(&mut output, validated.dimension)?;
        push_u64(&mut output, validated.target_size)?;
        push_u64(&mut output, validated.vectors.len())?;
        for vector in &validated.vectors {
            push_u64(&mut output, vector.len())?;
            for value in vector {
                let re = value.re().to_f64().ok_or_else(|| {
                    checkpoint_error("real component cannot be represented as f64")
                })?;
                let im = value.im().to_f64().ok_or_else(|| {
                    checkpoint_error("imaginary component cannot be represented as f64")
                })?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(checkpoint_error("checkpoint contains non-finite value"));
                }
                output.extend_from_slice(&re.to_le_bytes());
                output.extend_from_slice(&im.to_le_bytes());
            }
        }
        Ok(output)
    }

    /// Restore a checkpoint produced by [`Self::to_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = CheckpointReader { bytes, offset: 0 };
        if reader.take(8)? != CHECKPOINT_MAGIC {
            return Err(checkpoint_error("invalid GCRO-DR checkpoint magic"));
        }
        if reader.read_u16()? != GCRO_DR_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error("unsupported GCRO-DR checkpoint version"));
        }
        if reader.read_u8()? != std::mem::size_of::<T>() as u8
            || reader.read_u8()? != std::mem::size_of::<T::Real>() as u8
            || reader.read_u8()? != scalar_is_complex::<T>()
        {
            return Err(checkpoint_error("checkpoint scalar family does not match"));
        }
        let _reserved = reader.read_u8()?;
        let dimension = reader.read_usize()?;
        let target_size = reader.read_usize()?;
        let vector_count = reader.read_usize()?;
        if vector_count > target_size {
            return Err(checkpoint_error(
                "checkpoint vector count exceeds target size",
            ));
        }
        let mut vectors = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
            let length = reader.read_usize()?;
            if length != dimension {
                return Err(checkpoint_error(
                    "checkpoint vector dimension does not match header",
                ));
            }
            let mut vector = Vec::with_capacity(length);
            for _ in 0..length {
                let re = reader.read_f64()?;
                let im = reader.read_f64()?;
                let re = T::Real::from_f64(re)
                    .ok_or_else(|| checkpoint_error("real component cannot be restored"))?;
                let im = T::Real::from_f64(im)
                    .ok_or_else(|| checkpoint_error("imaginary component cannot be restored"))?;
                let value = T::from_re_im(re, im);
                if !value.norm().is_finite() {
                    return Err(checkpoint_error("checkpoint restores a non-finite value"));
                }
                vector.push(value);
            }
            vectors.push(vector);
        }
        if reader.offset != bytes.len() {
            return Err(checkpoint_error("trailing bytes in GCRO-DR checkpoint"));
        }
        let checkpoint = Self::from_vectors(target_size, &vectors)?;
        if checkpoint.dimension != dimension {
            return Err(checkpoint_error(
                "checkpoint vector dimension does not match header",
            ));
        }
        Ok(checkpoint)
    }
}

fn scalar_is_complex<T: ComplexField>() -> u8 {
    u8::from(T::from_re_im(T::Real::zero(), T::Real::one()).norm() > T::Real::zero())
}

fn checkpoint_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "GCRO-DR checkpoint",
        reason: reason.into(),
    }
}

fn push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value =
        u64::try_from(value).map_err(|_| checkpoint_error("checkpoint size does not fit u64"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

struct CheckpointReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CheckpointReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| checkpoint_error("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(checkpoint_error("truncated GCRO-DR checkpoint"));
        }
        let slice = &self.bytes[self.offset..end];
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

    fn read_usize(&mut self) -> Result<usize, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        usize::try_from(u64::from_le_bytes(value))
            .map_err(|_| checkpoint_error("checkpoint size does not fit usize"))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let mut value = [0_u8; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(f64::from_le_bytes(value))
    }
}
