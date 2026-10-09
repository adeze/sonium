//! Portable checkpoints for deflated-GMRES subspaces.
//!
//! A deflated subspace contains operator-dependent derived state (`A * W` and
//! the coarse LU factor).  Only the original deflation vectors are portable;
//! resumed solves rebuild the derived state against their current operator.

use super::gmres::{GmresConfig, GmresSolution};
use super::gmres_deflated::{
    DeflationSubspace, try_gmres_deflated, try_gmres_deflated_preconditioned,
};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use sonium_solvers_core::traits::{ComplexField, LinearOperator, Preconditioner, SolverError};

/// Version of the portable deflated-GMRES checkpoint format.
pub const DEFLATED_CHECKPOINT_FORMAT_VERSION: u16 = 1;

const CHECKPOINT_MAGIC: &[u8; 8] = b"SDEFLCP1";

/// A validated, operator-independent deflated-GMRES subspace checkpoint.
#[derive(Debug, Clone)]
pub struct DeflatedCheckpoint<T: ComplexField> {
    dimension: usize,
    vectors: Vec<Vec<T>>,
}

impl<T: ComplexField> DeflatedCheckpoint<T> {
    /// Capture the original vectors from a constructed deflation subspace.
    pub fn from_subspace(subspace: &DeflationSubspace<T>) -> Result<Self, SolverError> {
        Self::from_vectors(subspace.vectors())
    }

    /// Build a checkpoint from explicit deflation vectors.
    pub fn from_vectors(vectors: &[Vec<T>]) -> Result<Self, SolverError> {
        let dimension = vectors.first().map_or(0, Vec::len);
        if vectors.iter().any(|vector| vector.len() != dimension) {
            return Err(checkpoint_error(
                "deflation vectors have inconsistent dimensions",
            ));
        }
        if vectors
            .iter()
            .flat_map(|vector| vector.iter())
            .any(|value| !value.norm().is_finite())
        {
            return Err(checkpoint_error(
                "deflation vector contains NaN or infinity",
            ));
        }
        if !vectors.is_empty() && dimension == 0 {
            return Err(checkpoint_error("deflation vectors must not be empty"));
        }
        Ok(Self {
            dimension,
            vectors: vectors.to_vec(),
        })
    }

    /// Number of entries in each stored vector, or zero for an empty rank.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of vectors in the checkpoint.
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

    /// Borrow the validated original deflation vectors.
    pub fn vectors(&self) -> &[Vec<T>] {
        &self.vectors
    }

    /// Rebuild the operator-dependent deflation state.
    pub fn into_subspace<A>(&self, operator: &A) -> Result<DeflationSubspace<T>, SolverError>
    where
        A: LinearOperator<T>,
    {
        if !self.vectors.is_empty() && operator.num_cols() != self.dimension {
            return Err(SolverError::DimensionMismatch {
                expected: operator.num_cols(),
                got: self.dimension,
            });
        }
        DeflationSubspace::try_new(self.vectors.clone(), operator)
    }

    /// Restore this checkpoint into an existing subspace for `operator`.
    pub fn restore_into<A>(
        &self,
        subspace: &mut DeflationSubspace<T>,
        operator: &A,
    ) -> Result<(), SolverError>
    where
        A: LinearOperator<T>,
    {
        *subspace = self.into_subspace(operator)?;
        Ok(())
    }

    /// Resume an identity-preconditioned deflated-GMRES solve.
    pub fn resume<A>(
        &self,
        operator: &A,
        rhs: &[T],
        initial_guess: Option<&[T]>,
        config: &GmresConfig<T::Real>,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
    {
        let subspace = self.into_subspace(operator)?;
        try_gmres_deflated(operator, &subspace, rhs, initial_guess, config)
    }

    /// Resume a left-preconditioned deflated-GMRES solve.
    pub fn resume_preconditioned<A, P>(
        &self,
        operator: &A,
        preconditioner: &P,
        rhs: &[T],
        initial_guess: Option<&[T]>,
        config: &GmresConfig<T::Real>,
    ) -> Result<GmresSolution<T>, SolverError>
    where
        A: LinearOperator<T>,
        P: Preconditioner<T>,
    {
        let subspace = self.into_subspace(operator)?;
        try_gmres_deflated_preconditioned(
            operator,
            preconditioner,
            &subspace,
            rhs,
            initial_guess,
            config,
        )
    }

    /// Serialize the checkpoint as a strict, versioned little-endian format.
    /// Numeric values are canonical f64 real/imaginary pairs.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let validated = Self::from_vectors(&self.vectors)?;
        let mut output = Vec::with_capacity(32 + validated.storage_bytes());
        output.extend_from_slice(CHECKPOINT_MAGIC);
        output.extend_from_slice(&DEFLATED_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        output.push(u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX));
        output.push(scalar_is_complex::<T>());
        output.extend_from_slice(&0_u16.to_le_bytes());
        push_u64(&mut output, validated.dimension)?;
        push_u64(&mut output, validated.vectors.len())?;
        for vector in &validated.vectors {
            for value in vector {
                let re = value
                    .re()
                    .to_f64()
                    .ok_or_else(|| checkpoint_error("real component cannot convert to f64"))?;
                let im = value
                    .im()
                    .to_f64()
                    .ok_or_else(|| checkpoint_error("imaginary component cannot convert to f64"))?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(checkpoint_error("checkpoint contains NaN or infinity"));
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
        if reader.take(CHECKPOINT_MAGIC.len())? != CHECKPOINT_MAGIC {
            return Err(checkpoint_error("invalid checkpoint magic"));
        }
        if reader.read_u16()? != DEFLATED_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error("unsupported checkpoint version"));
        }
        let scalar_width = reader.read_u8()?;
        let expected_width = u8::try_from(std::mem::size_of::<T::Real>()).unwrap_or(u8::MAX);
        if scalar_width != expected_width {
            return Err(checkpoint_error("checkpoint scalar width mismatch"));
        }
        if reader.read_u8()? != scalar_is_complex::<T>() {
            return Err(checkpoint_error("checkpoint scalar family mismatch"));
        }
        if reader.read_u16()? != 0 {
            return Err(checkpoint_error("checkpoint reserved header is non-zero"));
        }
        let dimension = reader.read_usize()?;
        let rank = reader.read_usize()?;
        if (rank > 0 && dimension == 0) || (rank == 0 && dimension != 0) {
            return Err(checkpoint_error("checkpoint has invalid vector dimension"));
        }
        let value_count = rank
            .checked_mul(dimension)
            .ok_or_else(|| checkpoint_error("checkpoint vector count overflows usize"))?;
        let byte_count = value_count
            .checked_mul(16)
            .ok_or_else(|| checkpoint_error("checkpoint payload size overflows usize"))?;
        if reader.remaining() != byte_count {
            return Err(checkpoint_error("checkpoint payload length mismatch"));
        }
        let mut vectors = Vec::with_capacity(rank);
        for _ in 0..rank {
            let mut vector = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let re = reader.read_f64()?;
                let im = reader.read_f64()?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(checkpoint_error("checkpoint contains NaN or infinity"));
                }
                let re = T::Real::from_f64(re)
                    .ok_or_else(|| checkpoint_error("real component conversion failed"))?;
                let im = T::Real::from_f64(im)
                    .ok_or_else(|| checkpoint_error("imaginary component conversion failed"))?;
                if !re.is_finite() || !im.is_finite() {
                    return Err(checkpoint_error(
                        "checkpoint component is not representable",
                    ));
                }
                vector.push(T::from_re_im(re, im));
            }
            vectors.push(vector);
        }
        let checkpoint = Self::from_vectors(&vectors)?;
        if checkpoint.dimension != dimension {
            return Err(checkpoint_error(
                "checkpoint vector dimension does not match header",
            ));
        }
        Ok(checkpoint)
    }
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
            return Err(checkpoint_error("checkpoint is truncated"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, SolverError> {
        let bytes = self.take(2)?;
        let value = [bytes[0], bytes[1]];
        Ok(u16::from_le_bytes(value))
    }

    fn read_usize(&mut self) -> Result<usize, SolverError> {
        let bytes = self.take(8)?;
        let value: [u8; 8] = bytes
            .try_into()
            .map_err(|_| checkpoint_error("checkpoint size field is truncated"))?;
        usize::try_from(u64::from_le_bytes(value))
            .map_err(|_| checkpoint_error("checkpoint size does not fit usize"))
    }

    fn read_f64(&mut self) -> Result<f64, SolverError> {
        let bytes = self.take(8)?;
        let value: [u8; 8] = bytes
            .try_into()
            .map_err(|_| checkpoint_error("checkpoint scalar field is truncated"))?;
        Ok(f64::from_le_bytes(value))
    }
}

fn scalar_is_complex<T: ComplexField>() -> u8 {
    let one = T::Real::from_f64(1.0).unwrap_or_else(T::Real::zero);
    u8::from(T::from_re_im(T::Real::zero(), one).norm() > T::Real::zero())
}

fn checkpoint_error(reason: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "deflated GMRES checkpoint",
        reason: reason.into(),
    }
}

fn push_u64(output: &mut Vec<u8>, value: usize) -> Result<(), SolverError> {
    let value =
        u64::try_from(value).map_err(|_| checkpoint_error("checkpoint size does not fit u64"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}
