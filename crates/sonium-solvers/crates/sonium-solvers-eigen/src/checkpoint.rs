//! Strict portable checkpoints for native eigen-solver result states.

use num_complex::Complex64;
use sonium_solvers_core::{dense::DenseMatrix, traits::SolverError};

use crate::{ComplexTargetedEigenResult, LobpcgResult, TargetedEigenResult};

/// Native eigen-result checkpoint format version.
pub const EIGEN_CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Magic prefix shared by native real, complex, and LOBPCG result packets.
pub const EIGEN_CHECKPOINT_MAGIC: [u8; 8] = *b"EIGCP001";

const KIND_TARGETED_REAL: u8 = 1;
const KIND_TARGETED_COMPLEX: u8 = 2;
const KIND_LOBPCG: u8 = 3;
const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

impl TargetedEigenResult {
    /// Serialize a finite real shift-invert result.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate_real(&self.eigenvalues, &self.eigenvectors, &self.residual_norms)?;
        let mut bytes = Vec::new();
        write_header(
            &mut bytes,
            KIND_TARGETED_REAL,
            self.iterations,
            self.converged,
        );
        write_real_vector(&mut bytes, &self.eigenvalues);
        write_real_matrix(&mut bytes, &self.eigenvectors);
        write_real_vector(&mut bytes, &self.residual_norms);
        Ok(bytes)
    }

    /// Restore a real shift-invert result from a native checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        let (iterations, converged) = reader.header(KIND_TARGETED_REAL)?;
        let eigenvalues = reader.real_vector()?;
        let eigenvectors = reader.real_matrix()?;
        let residual_norms = reader.real_vector()?;
        reader.finish()?;
        validate_real(&eigenvalues, &eigenvectors, &residual_norms)?;
        Ok(Self {
            eigenvalues,
            eigenvectors,
            residual_norms,
            iterations,
            converged,
        })
    }
}

impl LobpcgResult {
    /// Serialize a finite LOBPCG result.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate_real(&self.eigenvalues, &self.eigenvectors, &self.residual_norms)?;
        let mut bytes = Vec::new();
        write_header(&mut bytes, KIND_LOBPCG, self.iterations, self.converged);
        write_real_vector(&mut bytes, &self.eigenvalues);
        write_real_matrix(&mut bytes, &self.eigenvectors);
        write_real_vector(&mut bytes, &self.residual_norms);
        Ok(bytes)
    }

    /// Restore a LOBPCG result from a native checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        let (iterations, converged) = reader.header(KIND_LOBPCG)?;
        let eigenvalues = reader.real_vector()?;
        let eigenvectors = reader.real_matrix()?;
        let residual_norms = reader.real_vector()?;
        reader.finish()?;
        validate_real(&eigenvalues, &eigenvectors, &residual_norms)?;
        Ok(Self {
            eigenvalues,
            eigenvectors,
            iterations,
            residual_norms,
            converged,
        })
    }
}

impl ComplexTargetedEigenResult {
    /// Serialize a finite complex Arnoldi result.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        validate_complex(&self.eigenvalues, &self.eigenvectors, &self.residual_norms)?;
        let mut bytes = Vec::new();
        write_header(
            &mut bytes,
            KIND_TARGETED_COMPLEX,
            self.iterations,
            self.converged,
        );
        write_complex_vector(&mut bytes, &self.eigenvalues);
        write_complex_matrix(&mut bytes, &self.eigenvectors);
        write_real_vector(&mut bytes, &self.residual_norms);
        Ok(bytes)
    }

    /// Restore a complex Arnoldi result from a native checkpoint.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        let (iterations, converged) = reader.header(KIND_TARGETED_COMPLEX)?;
        let eigenvalues = reader.complex_vector()?;
        let eigenvectors = reader.complex_matrix()?;
        let residual_norms = reader.real_vector()?;
        reader.finish()?;
        validate_complex(&eigenvalues, &eigenvectors, &residual_norms)?;
        Ok(Self {
            eigenvalues,
            eigenvectors,
            residual_norms,
            iterations,
            converged,
        })
    }
}

fn checkpoint_error(reason: &str) -> SolverError {
    SolverError::OperationFailed {
        operation: "eigen checkpoint",
        reason: reason.into(),
    }
}

fn validate_real(
    eigenvalues: &[f64],
    eigenvectors: &DenseMatrix<f64>,
    residual_norms: &[f64],
) -> Result<(), SolverError> {
    if eigenvalues.len() != residual_norms.len()
        || eigenvectors.ncols() != eigenvalues.len()
        || eigenvectors
            .as_slice()
            .iter()
            .any(|value| !value.is_finite())
        || eigenvalues.iter().any(|value| !value.is_finite())
        || residual_norms
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(checkpoint_error(
            "non-finite values or inconsistent result shape",
        ));
    }
    Ok(())
}

fn validate_complex(
    eigenvalues: &[Complex64],
    eigenvectors: &DenseMatrix<Complex64>,
    residual_norms: &[f64],
) -> Result<(), SolverError> {
    if eigenvalues.len() != residual_norms.len()
        || eigenvectors.ncols() != eigenvalues.len()
        || eigenvalues
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || eigenvectors
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || residual_norms
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(checkpoint_error(
            "non-finite values or inconsistent result shape",
        ));
    }
    Ok(())
}

fn write_header(bytes: &mut Vec<u8>, kind: u8, iterations: usize, converged: bool) {
    bytes.extend_from_slice(&EIGEN_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&EIGEN_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    bytes.push(kind);
    bytes.push(u8::from(converged));
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&u64::try_from(iterations).unwrap_or(u64::MAX).to_le_bytes());
}

fn write_real_vector(bytes: &mut Vec<u8>, values: &[f64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn write_complex_vector(bytes: &mut Vec<u8>, values: &[Complex64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.re.to_le_bytes());
        bytes.extend_from_slice(&value.im.to_le_bytes());
    }
}

fn write_real_matrix(bytes: &mut Vec<u8>, matrix: &DenseMatrix<f64>) {
    bytes.extend_from_slice(&(matrix.nrows() as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.ncols() as u64).to_le_bytes());
    for value in matrix.as_slice() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn write_complex_matrix(bytes: &mut Vec<u8>, matrix: &DenseMatrix<Complex64>) {
    bytes.extend_from_slice(&(matrix.nrows() as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.ncols() as u64).to_le_bytes());
    for value in matrix.as_slice() {
        bytes.extend_from_slice(&value.re.to_le_bytes());
        bytes.extend_from_slice(&value.im.to_le_bytes());
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
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("checked length"),
        ))
    }

    fn u32(&mut self) -> Result<u32, SolverError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("checked length"),
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

    fn header(&mut self, expected_kind: u8) -> Result<(usize, bool), SolverError> {
        if self.take(EIGEN_CHECKPOINT_MAGIC.len())? != EIGEN_CHECKPOINT_MAGIC {
            return Err(checkpoint_error("invalid magic"));
        }
        if self.u16()? != EIGEN_CHECKPOINT_FORMAT_VERSION {
            return Err(checkpoint_error("unsupported checkpoint version"));
        }
        if self.take(1)?[0] != expected_kind {
            return Err(checkpoint_error("checkpoint result kind mismatch"));
        }
        let flags = self.take(1)?[0];
        if flags & !1 != 0 || self.u32()? != 0 {
            return Err(checkpoint_error("reserved flags are non-zero"));
        }
        let iterations = usize::try_from(self.u64()?)
            .map_err(|_| checkpoint_error("iteration count overflows usize"))?;
        Ok((iterations, flags & 1 != 0))
    }

    fn length(&mut self, item_width: usize) -> Result<usize, SolverError> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| checkpoint_error("vector length overflows usize"))?;
        if length > MAX_ELEMENTS
            || length > self.bytes.len().saturating_sub(self.offset) / item_width
        {
            return Err(checkpoint_error("vector length is unreasonable"));
        }
        Ok(length)
    }

    fn real_vector(&mut self) -> Result<Vec<f64>, SolverError> {
        let length = self.length(8)?;
        (0..length).map(|_| self.f64()).collect()
    }

    fn complex_vector(&mut self) -> Result<Vec<Complex64>, SolverError> {
        let length = self.length(16)?;
        (0..length)
            .map(|_| Ok(Complex64::new(self.f64()?, self.f64()?)))
            .collect()
    }

    fn real_matrix(&mut self) -> Result<DenseMatrix<f64>, SolverError> {
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

    fn complex_matrix(&mut self) -> Result<DenseMatrix<Complex64>, SolverError> {
        let rows = self.length(1)?;
        let cols = usize::try_from(self.u64()?)
            .map_err(|_| checkpoint_error("matrix columns overflows usize"))?;
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| checkpoint_error("matrix shape overflows usize"))?;
        if count > MAX_ELEMENTS || count > self.bytes.len().saturating_sub(self.offset) / 16 {
            return Err(checkpoint_error("matrix shape is unreasonable"));
        }
        let values: Vec<Complex64> = (0..count)
            .map(|_| Ok(Complex64::new(self.f64()?, self.f64()?)))
            .collect::<Result<_, SolverError>>()?;
        Ok(DenseMatrix::from_row_major_vec(rows, cols, values))
    }

    fn finish(&self) -> Result<(), SolverError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(checkpoint_error("trailing bytes"))
        }
    }
}
