//! Native checkpoint codec for block FGMRES result state.

use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AccuracyStatus, SolverError},
};

use super::{BlockFgmresSolution, KrylovOpCounts};

/// Block-FGMRES checkpoint format version.
pub const BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Magic prefix for native f64 block-FGMRES result packets.
pub const BLOCK_FGMRES_CHECKPOINT_MAGIC: [u8; 8] = *b"BLKCP001";

const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

impl BlockFgmresSolution<f64> {
    /// Serialize a finite block-FGMRES result with strict metadata checks.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate(self)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BLOCK_FGMRES_CHECKPOINT_MAGIC);
        bytes.extend_from_slice(&BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        bytes.push(u8::from(self.converged));
        bytes.push(0); // reserved flags
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&(self.iterations as u64).to_le_bytes());
        write_matrix(&mut bytes, &self.x);
        write_vector(&mut bytes, &self.residuals);
        write_vector(&mut bytes, &self.true_residuals);
        write_vector(&mut bytes, &self.backward_errors);
        bytes.extend_from_slice(&(self.attainable_accuracy.len() as u64).to_le_bytes());
        for status in &self.attainable_accuracy {
            bytes.push(match status {
                AccuracyStatus::NotChecked => 0,
                AccuracyStatus::Attained => 1,
                AccuracyStatus::NotAttained => 2,
            });
        }
        for value in [
            self.op_counts.spmv_count as u64,
            self.op_counts.preconditioner_count as u64,
            self.op_counts.dot_reductions as u64,
            self.op_counts.norm_count as u64,
            self.op_counts.axpy_passes as u64,
            self.op_counts.basis_bytes_touched,
            self.op_counts.synchronization_points as u64,
            self.op_counts.exact_residual_checks as u64,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Restore a block-FGMRES result from a native checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        if reader.take(BLOCK_FGMRES_CHECKPOINT_MAGIC.len())? != BLOCK_FGMRES_CHECKPOINT_MAGIC {
            return Err(checkpoint_error("invalid magic"));
        }
        if reader.u16()? != BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error("unsupported checkpoint version"));
        }
        let flags = reader.take(1)?[0];
        if flags & !1 != 0 || reader.take(1)?[0] != 0 || reader.u16()? != 0 {
            return Err(checkpoint_error("reserved flags are non-zero"));
        }
        let converged = flags & 1 != 0;
        let iterations = usize::try_from(reader.u64()?)
            .map_err(|_| checkpoint_error("iteration count overflows usize"))?;
        let x = reader.matrix()?;
        let residuals = reader.vector()?;
        let true_residuals = reader.vector()?;
        let backward_errors = reader.vector()?;
        let accuracy_len = reader.length(1)?;
        let mut attainable_accuracy = Vec::with_capacity(accuracy_len);
        for _ in 0..accuracy_len {
            attainable_accuracy.push(match reader.take(1)?[0] {
                0 => AccuracyStatus::NotChecked,
                1 => AccuracyStatus::Attained,
                2 => AccuracyStatus::NotAttained,
                _ => return Err(checkpoint_error("invalid accuracy status")),
            });
        }
        let counts = [
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
        ];
        if !reader.is_empty() {
            return Err(checkpoint_error("trailing bytes"));
        }
        let op_counts = KrylovOpCounts {
            spmv_count: usize::try_from(counts[0])
                .map_err(|_| checkpoint_error("count overflow"))?,
            preconditioner_count: usize::try_from(counts[1])
                .map_err(|_| checkpoint_error("count overflow"))?,
            dot_reductions: usize::try_from(counts[2])
                .map_err(|_| checkpoint_error("count overflow"))?,
            norm_count: usize::try_from(counts[3])
                .map_err(|_| checkpoint_error("count overflow"))?,
            axpy_passes: usize::try_from(counts[4])
                .map_err(|_| checkpoint_error("count overflow"))?,
            basis_bytes_touched: counts[5],
            synchronization_points: usize::try_from(counts[6])
                .map_err(|_| checkpoint_error("count overflow"))?,
            exact_residual_checks: usize::try_from(counts[7])
                .map_err(|_| checkpoint_error("count overflow"))?,
        };
        let result = Self {
            x,
            iterations,
            residuals,
            true_residuals,
            backward_errors,
            attainable_accuracy,
            converged,
            op_counts,
        };
        validate(&result)?;
        Ok(result)
    }
}

fn checkpoint_error(reason: &str) -> SolverError {
    SolverError::OperationFailed {
        operation: "block FGMRES checkpoint",
        reason: reason.into(),
    }
}

fn validate(result: &BlockFgmresSolution<f64>) -> Result<(), SolverError> {
    let columns = result.x.ncols();
    if result.residuals.len() != columns
        || result.true_residuals.len() != columns
        || result.backward_errors.len() != columns
        || result.attainable_accuracy.len() != columns
        || result.x.as_slice().iter().any(|value| !value.is_finite())
        || result
            .residuals
            .iter()
            .chain(&result.true_residuals)
            .chain(&result.backward_errors)
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(checkpoint_error(
            "non-finite values or inconsistent result shape",
        ));
    }
    Ok(())
}

fn write_vector(bytes: &mut Vec<u8>, values: &[f64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn write_matrix(bytes: &mut Vec<u8>, matrix: &DenseMatrix<f64>) {
    bytes.extend_from_slice(&(matrix.nrows() as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.ncols() as u64).to_le_bytes());
    for value in matrix.as_slice() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
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
            .ok_or_else(|| checkpoint_error("packet offset overflow"))?;
        if end > self.bytes.len() {
            return Err(checkpoint_error("truncated packet"));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
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

    fn f64(&mut self) -> Result<f64, SolverError> {
        Ok(f64::from_le_bytes(
            self.take(8)?.try_into().expect("checked length"),
        ))
    }

    fn length(&mut self, width: usize) -> Result<usize, SolverError> {
        let length =
            usize::try_from(self.u64()?).map_err(|_| checkpoint_error("length overflows usize"))?;
        if length > MAX_ELEMENTS || length > self.bytes.len().saturating_sub(self.offset) / width {
            return Err(checkpoint_error("length is unreasonable"));
        }
        Ok(length)
    }

    fn vector(&mut self) -> Result<Vec<f64>, SolverError> {
        let length = self.length(8)?;
        (0..length).map(|_| self.f64()).collect()
    }

    fn matrix(&mut self) -> Result<DenseMatrix<f64>, SolverError> {
        let rows = self.length(1)?;
        let cols = usize::try_from(self.u64()?)
            .map_err(|_| checkpoint_error("matrix columns overflows usize"))?;
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| checkpoint_error("matrix shape overflows usize"))?;
        if count > MAX_ELEMENTS || count > self.bytes.len().saturating_sub(self.offset) / 8 {
            return Err(checkpoint_error("matrix shape is unreasonable"));
        }
        let values: Vec<f64> = (0..count).map(|_| self.f64()).collect::<Result<_, _>>()?;
        Ok(DenseMatrix::from_row_major_vec(rows, cols, values))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
