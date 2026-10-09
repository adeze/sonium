//! Ndarray-free host containers shared by upper Sonium crates.
//!
//! These types intentionally cover only the simple contiguous layouts used by
//! BEM, FEM, wave simulation, and QA fixtures during the migration away from
//! `ndarray`. Numerical kernels should still operate on slices.

use num_traits::Zero;
use std::ops::{Index, IndexMut};

/// Row-major 2D host matrix backed by a contiguous `Vec<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> HostMatrix<T> {
    pub fn from_elem(rows: usize, cols: usize, value: T) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn from_row_major_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "HostMatrix data length mismatch");
        Self { rows, cols, data }
    }

    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }
        Self { rows, cols, data }
    }
}

impl<T: Clone + Zero> HostMatrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::from_elem(rows, cols, T::zero())
    }
}

impl<T> HostMatrix<T> {
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    #[inline]
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows, "HostMatrix row out of bounds");
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.rows, "HostMatrix row out of bounds");
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    #[inline]
    fn offset(&self, row: usize, col: usize) -> usize {
        debug_assert!(
            row < self.rows && col < self.cols,
            "HostMatrix index out of bounds"
        );
        row * self.cols + col
    }
}

impl<T: Clone> HostMatrix<T> {
    pub fn column_to_vec(&self, col: usize) -> Vec<T> {
        assert!(col < self.cols, "HostMatrix column out of bounds");
        (0..self.rows)
            .map(|row| self.data[row * self.cols + col].clone())
            .collect()
    }

    pub fn set_column(&mut self, col: usize, values: &[T]) {
        assert!(col < self.cols, "HostMatrix column out of bounds");
        assert_eq!(values.len(), self.rows, "HostMatrix column length mismatch");
        for (row, value) in values.iter().enumerate() {
            self.data[row * self.cols + col] = value.clone();
        }
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T> Index<[usize; 2]> for HostMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[self.offset(index[0], index[1])]
    }
}

impl<T> IndexMut<[usize; 2]> for HostMatrix<T> {
    #[inline]
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let offset = self.offset(index[0], index[1]);
        &mut self.data[offset]
    }
}

impl<T> Index<(usize, usize)> for HostMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.offset(index.0, index.1)]
    }
}

impl<T> IndexMut<(usize, usize)> for HostMatrix<T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let offset = self.offset(index.0, index.1);
        &mut self.data[offset]
    }
}

/// Row-major 3D host tensor backed by a contiguous `Vec<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostTensor3<T> {
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<T>,
}

impl<T: Clone> HostTensor3<T> {
    pub fn from_elem(shape: (usize, usize, usize), value: T) -> Self {
        let (nx, ny, nz) = shape;
        Self {
            nx,
            ny,
            nz,
            data: vec![value; nx * ny * nz],
        }
    }

    pub fn from_row_major_vec(shape: (usize, usize, usize), data: Vec<T>) -> Self {
        let (nx, ny, nz) = shape;
        assert_eq!(data.len(), nx * ny * nz, "HostTensor3 data length mismatch");
        Self { nx, ny, nz, data }
    }

    pub fn from_fn<F>(shape: (usize, usize, usize), mut f: F) -> Self
    where
        F: FnMut(usize, usize, usize) -> T,
    {
        let (nx, ny, nz) = shape;
        let mut data = Vec::with_capacity(nx * ny * nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    data.push(f(i, j, k));
                }
            }
        }
        Self { nx, ny, nz, data }
    }
}

impl<T: Clone + Zero> HostTensor3<T> {
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self::from_elem(shape, T::zero())
    }
}

impl<T> HostTensor3<T> {
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    #[inline]
    fn offset(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(
            i < self.nx && j < self.ny && k < self.nz,
            "HostTensor3 index out of bounds"
        );
        (i * self.ny + j) * self.nz + k
    }
}

impl<T> Index<[usize; 3]> for HostTensor3<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; 3]) -> &Self::Output {
        &self.data[self.offset(index[0], index[1], index[2])]
    }
}

impl<T> IndexMut<[usize; 3]> for HostTensor3<T> {
    #[inline]
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        let offset = self.offset(index[0], index[1], index[2]);
        &mut self.data[offset]
    }
}

impl<T> Index<(usize, usize, usize)> for HostTensor3<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.data[self.offset(index.0, index.1, index.2)]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for HostTensor3<T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let offset = self.offset(index.0, index.1, index.2);
        &mut self.data[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::{HostMatrix, HostTensor3};

    #[test]
    fn matrix_indexes_row_major() {
        let mut matrix = HostMatrix::from_fn(2, 3, |i, j| 10 * i + j);
        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix.as_slice(), &[0, 1, 2, 10, 11, 12]);
        assert_eq!(matrix[[1, 2]], 12);
        matrix[(0, 1)] = 7;
        assert_eq!(matrix.row(0), &[0, 7, 2]);
    }

    #[test]
    fn matrix_column_roundtrip() {
        let mut matrix = HostMatrix::zeros(3, 2);
        matrix.set_column(1, &[4, 5, 6]);
        assert_eq!(matrix.column_to_vec(1), vec![4, 5, 6]);
        assert_eq!(matrix.as_slice(), &[0, 4, 0, 5, 0, 6]);
    }

    #[test]
    fn tensor3_indexes_row_major() {
        let tensor = HostTensor3::from_fn((2, 2, 3), |i, j, k| 100 * i + 10 * j + k);
        assert_eq!(tensor.shape(), (2, 2, 3));
        assert_eq!(tensor[[1, 0, 2]], 102);
        assert_eq!(tensor[(1, 1, 2)], 112);
        assert_eq!(tensor.as_slice()[tensor.len() - 1], 112);
    }
}
