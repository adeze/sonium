//! Borrowed host-side CSR matrix view.
//!
//! This is a lightweight interchange shape for code that needs CSR metadata
//! before uploading to a backend. It deliberately does not own storage or know
//! about solver-side matrix types.

use thiserror::Error;

use crate::DeviceScalar;

/// Borrowed host-side CSR matrix.
#[derive(Debug, Clone, Copy)]
pub struct HostCsrRef<'a, T> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_ptrs: &'a [usize],
    pub col_indices: &'a [usize],
    pub values: &'a [T],
}

/// Errors returned by checked host-CSR construction and transformation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostCsrError {
    #[error("row_ptrs must have num_rows + 1 entries")]
    RowPointerCount,
    #[error("col_indices and values must have matching lengths")]
    MismatchedNnz,
    #[error("row_ptrs[num_rows] must equal nnz")]
    NnzMarker,
    #[error("row_ptrs must be monotonically nondecreasing")]
    NonMonotoneRows,
    #[error("CSR column index out of bounds")]
    ColumnOutOfBounds,
    #[error("CSR dimensions overflow host index space")]
    DimensionOverflow,
}

/// Owned CSR storage returned by checked sparse transformations.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedHostCsr<T> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
}

impl<T> OwnedHostCsr<T> {
    /// Borrow the owned matrix using the same interchange view as inputs.
    pub fn as_ref(&self) -> HostCsrRef<'_, T> {
        HostCsrRef {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            row_ptrs: &self.row_ptrs,
            col_indices: &self.col_indices,
            values: &self.values,
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

impl<'a, T> HostCsrRef<'a, T> {
    /// Construct a checked borrowed CSR view.
    pub fn try_new(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: &'a [usize],
        col_indices: &'a [usize],
        values: &'a [T],
    ) -> Result<Self, HostCsrError> {
        let expected_row_ptrs = num_rows
            .checked_add(1)
            .ok_or(HostCsrError::DimensionOverflow)?;
        if row_ptrs.len() != expected_row_ptrs {
            return Err(HostCsrError::RowPointerCount);
        }
        if col_indices.len() != values.len() {
            return Err(HostCsrError::MismatchedNnz);
        }
        if row_ptrs[num_rows] != values.len() {
            return Err(HostCsrError::NnzMarker);
        }
        if row_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(HostCsrError::NonMonotoneRows);
        }
        if col_indices.iter().any(|&column| column >= num_cols) {
            return Err(HostCsrError::ColumnOutOfBounds);
        }
        Ok(Self {
            num_rows,
            num_cols,
            row_ptrs,
            col_indices,
            values,
        })
    }

    pub fn new(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: &'a [usize],
        col_indices: &'a [usize],
        values: &'a [T],
    ) -> Self {
        assert_eq!(
            row_ptrs.len(),
            num_rows + 1,
            "row_ptrs must have num_rows + 1 entries"
        );
        assert_eq!(
            col_indices.len(),
            values.len(),
            "col_indices and values must have matching lengths"
        );
        assert_eq!(
            row_ptrs[num_rows],
            values.len(),
            "row_ptrs[num_rows] must equal nnz"
        );

        for window in row_ptrs.windows(2) {
            assert!(
                window[0] <= window[1],
                "row_ptrs must be monotonically nondecreasing"
            );
        }
        for &col in col_indices {
            assert!(col < num_cols, "CSR column index out of bounds");
        }

        Self {
            num_rows,
            num_cols,
            row_ptrs,
            col_indices,
            values,
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn is_square(&self) -> bool {
        self.num_rows == self.num_cols
    }

    /// Build the deterministic CSR representation of `Aᵀ`.
    ///
    /// Entries in each transposed row retain their source-row order. This is
    /// suitable for repeated sparse adjoint products and does not conjugate
    /// values; callers solving `Aᴴ` should conjugate values explicitly.
    pub fn try_transpose(&self) -> Result<OwnedHostCsr<T>, HostCsrError>
    where
        T: Clone,
    {
        Self::try_new(
            self.num_rows,
            self.num_cols,
            self.row_ptrs,
            self.col_indices,
            self.values,
        )?;

        let mut counts = vec![0usize; self.num_cols];
        for &column in self.col_indices {
            counts[column] += 1;
        }
        let row_pointer_count = self
            .num_cols
            .checked_add(1)
            .ok_or(HostCsrError::DimensionOverflow)?;
        let mut row_ptrs = vec![0usize; row_pointer_count];
        for row in 0..self.num_cols {
            row_ptrs[row + 1] = row_ptrs[row] + counts[row];
        }
        let mut next = row_ptrs[..self.num_cols].to_vec();
        let mut col_indices = vec![0usize; self.values.len()];
        let mut values: Vec<Option<T>> = (0..self.values.len()).map(|_| None).collect();

        for row in 0..self.num_rows {
            for entry in self.row_ptrs[row]..self.row_ptrs[row + 1] {
                let transposed_row = self.col_indices[entry];
                let destination = next[transposed_row];
                next[transposed_row] += 1;
                col_indices[destination] = row;
                values[destination] = Some(self.values[entry].clone());
            }
        }

        let values = values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(HostCsrError::MismatchedNnz)?;
        Ok(OwnedHostCsr {
            num_rows: self.num_cols,
            num_cols: self.num_rows,
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Build the deterministic CSR representation of the Hermitian adjoint
    /// `Aᴴ`.
    ///
    /// The sparsity transformation is identical to [`Self::try_transpose`],
    /// followed by elementwise conjugation. Real-valued matrices therefore
    /// receive the ordinary transpose.
    pub fn try_hermitian(&self) -> Result<OwnedHostCsr<T>, HostCsrError>
    where
        T: DeviceScalar,
    {
        let mut hermitian = self.try_transpose()?;
        for value in &mut hermitian.values {
            *value = value.conj_mul(T::one());
        }
        Ok(hermitian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_exposes_shape() {
        let row_ptrs = [0, 2, 3];
        let col_indices = [0, 1, 1];
        let values = [2.0_f64, -1.0, 3.0];
        let csr = HostCsrRef::new(2, 2, &row_ptrs, &col_indices, &values);

        assert_eq!(csr.num_rows, 2);
        assert_eq!(csr.num_cols, 2);
        assert_eq!(csr.nnz(), 3);
        assert!(csr.is_square());
    }

    #[test]
    fn checked_transpose_preserves_deterministic_entry_order() {
        let row_ptrs = [0, 1, 3];
        let col_indices = [2, 0, 2];
        let values = [5.0_f64, 6.0, 7.0];
        let csr = HostCsrRef::try_new(2, 3, &row_ptrs, &col_indices, &values)
            .expect("valid rectangular CSR");
        let transposed = csr.try_transpose().expect("transpose should succeed");
        assert_eq!(transposed.num_rows, 3);
        assert_eq!(transposed.num_cols, 2);
        assert_eq!(transposed.row_ptrs, [0, 1, 1, 3]);
        assert_eq!(transposed.col_indices, [1, 0, 1]);
        assert_eq!(transposed.values, [6.0, 5.0, 7.0]);
        assert_eq!(transposed.as_ref().nnz(), 3);
    }

    #[test]
    fn checked_hermitian_transposes_and_conjugates_complex_values() {
        use num_complex::Complex64;

        let row_ptrs = [0, 2, 3];
        let col_indices = [0, 2, 1];
        let values = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(-5.0, 6.0),
        ];
        let matrix = HostCsrRef::try_new(2, 3, &row_ptrs, &col_indices, &values).unwrap();
        let hermitian = matrix.try_hermitian().unwrap();

        assert_eq!(hermitian.num_rows, 3);
        assert_eq!(hermitian.num_cols, 2);
        assert_eq!(hermitian.row_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(hermitian.col_indices, vec![0, 1, 0]);
        assert_eq!(
            hermitian.values,
            vec![
                Complex64::new(1.0, -2.0),
                Complex64::new(-5.0, -6.0),
                Complex64::new(3.0, 4.0),
            ]
        );
    }

    #[test]
    fn checked_constructor_returns_typed_errors() {
        let error = HostCsrRef::try_new(1, 1, &[0, 2], &[0], &[1.0_f64])
            .expect_err("bad row marker must be rejected");
        assert_eq!(error, HostCsrError::NnzMarker);
    }

    #[test]
    #[should_panic(expected = "row_ptrs[num_rows] must equal nnz")]
    fn rejects_bad_nnz_marker() {
        let row_ptrs = [0, 1];
        let col_indices = [0, 0];
        let values = [1.0_f64, 2.0];
        let _ = HostCsrRef::new(1, 1, &row_ptrs, &col_indices, &values);
    }

    #[test]
    #[should_panic(expected = "CSR column index out of bounds")]
    fn rejects_out_of_range_columns() {
        let row_ptrs = [0, 1];
        let col_indices = [2];
        let values = [1.0_f64];
        let _ = HostCsrRef::new(1, 2, &row_ptrs, &col_indices, &values);
    }
}
