//! Minimal row-major dense matrix used internally by the solvers crate.
//!
//! Replaces `ndarray::Array2` for the in-crate use cases (LU factor storage,
//! Hessenberg matrices in GMRES, BFGS Hessians, Jacobians, etc). The surface
//! is deliberately small — just what call sites here actually need.
//!
//! Layout is row-major: element `(i, j)` lives at offset `i * ncols + j`.

use crate::traits::ComplexField;
use num_traits::Zero;
use rayon::prelude::*;
use sonium_compute::backend::Transpose;
use std::ops::{Index, IndexMut};

/// Minimum multiply-add count for [`DenseMatrix::matmul_into`] to split
/// output rows across rayon threads. Below it the pool dispatch plus the
/// per-panel B repack inside the tuned kernel exceed the gain; small
/// callers (LOBPCG Grams, tiny merge nodes) stay serial and bit-identical.
const PAR_GEMM_MIN_OPS: u64 = 2_000_000;

/// Minimum output rows per thread in a threaded [`DenseMatrix::matmul_into`]:
/// caps the panel count on wide pools so B-repack redundancy (one per panel)
/// stays negligible next to the per-panel compute.
const PAR_GEMM_MIN_ROWS: usize = 32;

/// Minimum output rows for any threaded GEMM entry (`matmul_into` and
/// `matmul_transa_into`): two MR-tiles worth. Below 64 rows the old
/// `2 * PAR_GEMM_MIN_ROWS` gate serialized small-m products with big n/k
/// (m = 32, n = k = 300: 5.8 M ops at ~35 GFLOPS serial); the finer
/// grain above keeps them parallel. The ops gate still holds, so tiny
/// callers stay serial.
const PAR_GEMM_MIN_M: usize = 8;

/// Row-major dense matrix backed by a contiguous `Vec<T>`.
#[derive(Debug, Clone)]
pub struct DenseMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

pub struct DenseColumn<'a, T> {
    matrix: &'a DenseMatrix<T>,
    col: usize,
}

impl<T: ComplexField> DenseColumn<'_, T> {
    /// Inner product `sum(conj(self[i]) * other[i])`, matching the
    /// [`ComplexField::vec_dot`] (`dotc`) contract. Conjugation is the
    /// identity for real scalars, so real callers are unaffected.
    pub fn dot(&self, other: &DenseColumn<'_, T>) -> T {
        assert_eq!(self.matrix.rows, other.matrix.rows);
        let mut acc = T::zero();
        for i in 0..self.matrix.rows {
            acc += self[i].conj() * other[i];
        }
        acc
    }

    pub fn to_owned(&self) -> Vec<T> {
        (0..self.matrix.rows).map(|i| self[i]).collect()
    }
}

impl<T> Index<usize> for DenseColumn<'_, T> {
    type Output = T;

    #[inline]
    fn index(&self, row: usize) -> &T {
        debug_assert!(row < self.matrix.rows, "column row out of bounds");
        &self.matrix.data[row * self.matrix.cols + self.col]
    }
}

pub struct DenseColumnMut<'a, T> {
    matrix: &'a mut DenseMatrix<T>,
    col: usize,
}

impl<T: Copy> DenseColumnMut<'_, T> {
    pub fn assign(&mut self, values: &[T]) {
        assert_eq!(values.len(), self.matrix.rows);
        for (row, value) in values.iter().enumerate() {
            self.matrix.data[row * self.matrix.cols + self.col] = *value;
        }
    }

    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        for row in 0..self.matrix.rows {
            let index = row * self.matrix.cols + self.col;
            self.matrix.data[index] = f(self.matrix.data[index]);
        }
    }
}

pub trait DenseDotRhs<T: ComplexField> {
    type Output;

    fn dot_with(self, lhs: &DenseMatrix<T>) -> Self::Output;
}

impl<T: ComplexField> DenseDotRhs<T> for &[T] {
    type Output = Vec<T>;

    fn dot_with(self, lhs: &DenseMatrix<T>) -> Self::Output {
        lhs.matvec(self)
    }
}

impl<T: ComplexField> DenseDotRhs<T> for &Vec<T> {
    type Output = Vec<T>;

    fn dot_with(self, lhs: &DenseMatrix<T>) -> Self::Output {
        lhs.matvec(self)
    }
}

impl<T: ComplexField> DenseDotRhs<T> for &DenseMatrix<T> {
    type Output = DenseMatrix<T>;

    fn dot_with(self, lhs: &DenseMatrix<T>) -> Self::Output {
        lhs.matmul(self)
    }
}

impl<T: Clone> DenseMatrix<T> {
    /// Construct an `r x c` matrix filled with `value`.
    pub fn from_elem(rows: usize, cols: usize, value: T) -> Self {
        let data = vec![value; rows * cols];
        Self { rows, cols, data }
    }

    /// Construct from a row-major `Vec<T>`. Panics on shape mismatch.
    pub fn from_row_major_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Self { rows, cols, data }
    }

    /// Build a matrix from a closure that produces `f(i, j)`.
    pub fn from_fn<F: FnMut(usize, usize) -> T>(rows: usize, cols: usize, mut f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }
}

impl<T: Clone + Zero> DenseMatrix<T> {
    /// Construct a zero matrix of size `rows x cols`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::from_elem(rows, cols, T::zero())
    }
}

impl<T: ComplexField> DenseMatrix<T> {
    /// Construct an `n x n` identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut m = Self::from_elem(n, n, T::zero());
        for i in 0..n {
            m[[i, i]] = T::one();
        }
        m
    }

    /// Construct a square matrix with `diag` on the diagonal.
    pub fn from_diag(diag: &[T]) -> Self {
        let n = diag.len();
        let mut m = Self::from_elem(n, n, T::zero());
        for i in 0..n {
            m[[i, i]] = diag[i];
        }
        m
    }
}

impl<T> DenseMatrix<T> {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Shape `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the row-major data slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the row-major data slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the matrix and return the underlying row-major `Vec<T>`.
    #[inline]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Take the underlying row-major `Vec<T>`, leaving a valid empty
    /// (0×0) matrix. Used by owner-level buffer pools (e.g. the LU factor
    /// pool) to recycle storage without moving the matrix.
    #[inline]
    pub fn take_data(&mut self) -> Vec<T> {
        self.rows = 0;
        self.cols = 0;
        std::mem::take(&mut self.data)
    }

    /// Borrow row `i` as a slice of length `ncols`.
    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Mutably borrow row `i` as a slice of length `ncols`.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

impl<T: Clone> DenseMatrix<T> {
    /// Copy column `j` into an owned `Vec<T>` of length `nrows`.
    pub fn column_to_vec(&self, j: usize) -> Vec<T> {
        assert!(j < self.cols);
        (0..self.rows)
            .map(|i| self.data[i * self.cols + j].clone())
            .collect()
    }

    /// Write column `j` from a slice of length `nrows`.
    pub fn set_column(&mut self, j: usize, col: &[T]) {
        assert!(j < self.cols);
        assert_eq!(col.len(), self.rows);
        for (i, value) in col.iter().enumerate() {
            self.data[i * self.cols + j] = value.clone();
        }
    }

    /// Swap rows `a` and `b`.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (start_a, start_b) = if a < b {
            (a * self.cols, b * self.cols)
        } else {
            (b * self.cols, a * self.cols)
        };
        let (left, right) = self.data.split_at_mut(start_b);
        let a_slice = &mut left[start_a..start_a + self.cols];
        let b_slice = &mut right[..self.cols];
        a_slice.swap_with_slice(b_slice);
    }
}

impl<T: Clone> DenseMatrix<T> {
    /// Fill every entry with `value`.
    pub fn fill(&mut self, value: T) {
        for v in self.data.iter_mut() {
            *v = value.clone();
        }
    }
}

impl<T> Index<[usize; 2]> for DenseMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: [usize; 2]) -> &T {
        let [i, j] = idx;
        debug_assert!(i < self.rows && j < self.cols, "index out of bounds");
        &self.data[i * self.cols + j]
    }
}

impl<T> IndexMut<[usize; 2]> for DenseMatrix<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        let [i, j] = idx;
        debug_assert!(i < self.rows && j < self.cols, "index out of bounds");
        &mut self.data[i * self.cols + j]
    }
}

impl<T> Index<(usize, usize)> for DenseMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: (usize, usize)) -> &T {
        let (i, j) = idx;
        debug_assert!(i < self.rows && j < self.cols, "index out of bounds");
        &self.data[i * self.cols + j]
    }
}

impl<T> IndexMut<(usize, usize)> for DenseMatrix<T> {
    #[inline]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        let (i, j) = idx;
        debug_assert!(i < self.rows && j < self.cols, "index out of bounds");
        &mut self.data[i * self.cols + j]
    }
}

impl<T: ComplexField> DenseMatrix<T> {
    /// `y = A * x` as `Vec<T>` (ndarray-compat alias for `matvec`).
    pub fn dot<R>(&self, rhs: R) -> R::Output
    where
        R: DenseDotRhs<T>,
    {
        rhs.dot_with(self)
    }

    /// Return the transpose as a new owned matrix.
    pub fn transposed(&self) -> DenseMatrix<T> {
        let mut out = DenseMatrix::from_elem(self.cols, self.rows, T::zero());
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[[j, i]] = self[[i, j]];
            }
        }
        out
    }

    /// Transpose alias (ndarray-compat).
    pub fn t(&self) -> DenseMatrix<T> {
        self.transposed()
    }

    /// Return the conjugate transpose as an owned matrix.
    ///
    /// Entry `(j, i)` of the result is the conjugate of entry `(i, j)`
    /// of `self`. Conjugation is the identity for real scalars, so on
    /// real matrices this matches [`Self::transposed`] exactly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use sonium_solvers_core::dense::DenseMatrix;
    ///
    /// let a = DenseMatrix::from_row_major_vec(
    ///     1,
    ///     2,
    ///     vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -3.0)],
    /// );
    /// let h = a.adjoint();
    /// assert_eq!(h.shape(), (2, 1));
    /// assert_eq!(h[[0, 0]], Complex64::new(1.0, -1.0));
    /// assert_eq!(h[[1, 0]], Complex64::new(2.0, 3.0));
    /// ```
    pub fn adjoint(&self) -> DenseMatrix<T> {
        let mut out = DenseMatrix::from_elem(self.cols, self.rows, T::zero());
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[[j, i]] = self[[i, j]].conj();
            }
        }
        out
    }

    /// Write column `j` from a Vec or iterator-like source. Wrapper to
    /// keep ndarray-style call sites working.
    pub fn column(&self, j: usize) -> DenseColumn<'_, T> {
        assert!(j < self.cols);
        DenseColumn {
            matrix: self,
            col: j,
        }
    }

    pub fn column_mut(&mut self, j: usize) -> DenseColumnMut<'_, T> {
        assert!(j < self.cols);
        DenseColumnMut {
            matrix: self,
            col: j,
        }
    }

    /// In-place column-write helper: `f(col_idx) -> new value`.
    pub fn column_assign_from<F: FnMut(usize) -> T>(&mut self, j: usize, mut f: F) {
        assert!(j < self.cols);
        for i in 0..self.rows {
            self.data[i * self.cols + j] = f(i);
        }
    }

    /// `y = A * x` (dense row-major matrix-vector product). Routes through
    /// `T::mat_vec_op`, which dispatches to `sonium_compute::cpu::blas2` for
    /// f32/f64 (SME2 on aarch64) and to `blas2::complex_native` (NEON /
    /// AVX2+FMA interleaved kernels) for Complex32/Complex64; the scalar
    /// default covers small panels and other `ComplexField` types like `f16`.
    pub fn matvec_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        T::mat_vec_op(Transpose::NoTrans, self.rows, self.cols, &self.data, x, y);
    }

    /// Returns `y = A * x` as an owned `Vec<T>`.
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.rows];
        self.matvec_into(x, &mut y);
        y
    }

    /// `C = A * B`. Routes through `T::mat_mul_nn`, which dispatches to
    /// `sonium_compute::cpu::blas3` for f32/f64/Complex32/Complex64 (SME2 on
    /// aarch64 when available). The default scalar triple loop covers other
    /// `ComplexField` types like `f16`.
    pub fn matmul(&self, rhs: &DenseMatrix<T>) -> DenseMatrix<T>
    where
        T: Send + Sync,
    {
        assert_eq!(self.cols, rhs.rows);
        let mut out = DenseMatrix::from_elem(self.rows, rhs.cols, T::zero());
        self.matmul_into(rhs, &mut out);
        out
    }

    /// `out = A * B`, writing into a pre-allocated `out`. Same
    /// `T::mat_mul_nn` dispatch as [`Self::matmul`]. Large products split
    /// output-row panels across rayon threads (each panel is a contiguous
    /// row range, so every panel call keeps the kernel's packed row-major
    /// contract); panels own disjoint `C` rows with the same `k`-loop order,
    /// hence results are bit-identical to the serial call at any thread
    /// count. Small products stay serial (see [`PAR_GEMM_MIN_OPS`]).
    pub fn matmul_into(&self, rhs: &DenseMatrix<T>, out: &mut DenseMatrix<T>)
    where
        T: Send + Sync,
    {
        assert_eq!(self.cols, rhs.rows);
        assert_eq!(out.rows, self.rows);
        assert_eq!(out.cols, rhs.cols);
        let (m, n, k) = (self.rows, rhs.cols, self.cols);
        let ops = m as u64 * n as u64 * k as u64;
        if ops >= PAR_GEMM_MIN_OPS && m >= 2 * PAR_GEMM_MIN_M {
            // Task grain targets ~16 row panels (the measured
            // bandwidth-saturation knee; more tasks only add dispatch +
            // B re-stream churn) with an MR-aligned floor of 4 so every
            // panel fills whole micro-tiles. Plain m/32 serializes
            // small-m products (m = 32 → 1 panel at ~35 GFLOPS instead of
            // ~90); the finer grain keeps them parallel. At scale the
            // grain saturates back at 32 (today's behavior) and panels at
            // the pool size. Bitwise-identical: disjoint C row panels
            // with the same k-loop order at any thread count.
            let target = rayon::current_num_threads().clamp(1, 16);
            let grain = (m / target).clamp(4, PAR_GEMM_MIN_ROWS);
            let panels = m
                .div_ceil(grain)
                .min(rayon::current_num_threads().max(1))
                .max(1);
            let rows_per = m.div_ceil(panels);
            let a_all = &self.data[..];
            let b_all = &rhs.data[..];
            out.data
                .par_chunks_mut(rows_per * n)
                .enumerate()
                .for_each(|(p, c_chunk)| {
                    let r0 = p * rows_per;
                    let rows = c_chunk.len() / n;
                    T::mat_mul_nn(rows, n, k, &a_all[r0 * k..(r0 + rows) * k], b_all, c_chunk);
                });
        } else {
            T::mat_mul_nn(m, n, k, &self.data, &rhs.data, &mut out.data);
        }
    }

    /// `C = Aᵀ * B` with `A` (`n × m`) = `self` and `B` (`n × k`) = `rhs`.
    /// Routes through `T::mat_mul_tn`, which keeps the transpose inside the
    /// tuned kernel (no `n × m` transpose allocation, unlike
    /// `self.t().matmul(rhs)`). Same result up to summation order.
    pub fn matmul_transa(&self, rhs: &DenseMatrix<T>) -> DenseMatrix<T> {
        assert_eq!(self.rows, rhs.rows);
        let mut out = DenseMatrix::from_elem(self.cols, rhs.cols, T::zero());
        self.matmul_transa_into(rhs, &mut out);
        out
    }

    /// `out = Aᵀ * B`, writing into a pre-allocated `out`. Same
    /// `T::mat_mul_tn` dispatch as [`Self::matmul_transa`].
    pub fn matmul_transa_into(&self, rhs: &DenseMatrix<T>, out: &mut DenseMatrix<T>) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(out.rows, self.cols);
        assert_eq!(out.cols, rhs.cols);
        // `C = Aᵀ·B`: A is `self.rows × self.cols`, C is
        // `self.cols × rhs.cols`. Row panels of C read column panels of
        // A, which are strided in row-major storage — but every kernel
        // derives `lda` from its `m` argument and requires tight
        // storage, so each panel packs its `k × rows` column window
        // into a scratch buffer first (row-segment memcpys, one per k
        // row). Same grain/entry as `matmul_into` (the transpose path
        // was serial-only before: 32×300×299 ran at ~30 GFLOPS).
        let (m, n, k) = (self.cols, rhs.cols, self.rows);
        let ops = m as u64 * n as u64 * k as u64;
        if ops >= PAR_GEMM_MIN_OPS && m >= 2 * PAR_GEMM_MIN_M {
            let target = rayon::current_num_threads().clamp(1, 16);
            let grain = (m / target).clamp(4, PAR_GEMM_MIN_ROWS);
            let panels = m
                .div_ceil(grain)
                .min(rayon::current_num_threads().max(1))
                .max(1);
            let rows_per = m.div_ceil(panels);
            let a_all = &self.data[..];
            let b_all = &rhs.data[..];
            out.data
                .par_chunks_mut(rows_per * n)
                .enumerate()
                .for_each(|(p, c_chunk)| {
                    let r0 = p * rows_per;
                    let rows = c_chunk.len() / n;
                    let mut apack = Vec::with_capacity(k * rows);
                    for pp in 0..k {
                        apack.extend_from_slice(&a_all[pp * m + r0..pp * m + r0 + rows]);
                    }
                    T::mat_mul_tn(rows, n, k, &apack, b_all, c_chunk);
                });
        } else {
            T::mat_mul_tn(
                self.cols,
                rhs.cols,
                self.rows,
                &self.data,
                &rhs.data,
                &mut out.data,
            );
        }
    }

    /// `y = A^T * x`. Same `T::mat_vec_op` dispatch as [`Self::matvec_into`].
    pub fn matvec_transpose_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.rows);
        assert_eq!(y.len(), self.cols);
        T::mat_vec_op(Transpose::Trans, self.rows, self.cols, &self.data, x, y);
    }

    /// `y = A^H * x` (conjugate transpose). Same `T::mat_vec_op` dispatch
    /// as [`Self::matvec_into`].
    pub fn matvec_hermitian_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.rows);
        assert_eq!(y.len(), self.cols);
        T::mat_vec_op(Transpose::ConjTrans, self.rows, self.cols, &self.data, x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_and_index() {
        let m = DenseMatrix::<f64>::eye(3);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[1, 1]], 1.0);
        assert_eq!(m[[2, 2]], 1.0);
        assert_eq!(m[[0, 1]], 0.0);
    }

    #[test]
    fn test_matvec() {
        // 2x3
        let m = DenseMatrix::from_row_major_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = vec![1.0, 0.0, -1.0];
        let y = m.matvec(&x);
        assert_eq!(y, vec![1.0 - 3.0, 4.0 - 6.0]);
    }

    #[test]
    fn column_dot_conjugates_left_factor() {
        use num_complex::Complex64;
        let a = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let b = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let d = a.column(0).dot(&b.column(0));
        // Matches `ComplexField::vec_dot`: conj(1+i)*(1-i) + 2*1 = 2-2i.
        let v = Complex64::vec_dot(&a.column(0).to_owned(), &b.column(0).to_owned());
        assert_eq!(d, v);
        assert_eq!(d, Complex64::new(2.0, -2.0));
    }

    #[test]
    fn test_swap_rows() {
        let mut m = DenseMatrix::from_row_major_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        m.swap_rows(0, 1);
        assert_eq!(m.row(0), &[4.0, 5.0, 6.0]);
        assert_eq!(m.row(1), &[1.0, 2.0, 3.0]);
    }

    // Deterministic pseudo-random value in [-1, 1) for parity tests.
    fn prng(k: usize) -> f64 {
        let h = k
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11;
        (h % 2000) as f64 / 1000.0 - 1.0
    }

    /// The blas2 kernels reorder summation relative to the scalar loop, so
    /// parity is asserted within a relative 2-norm tolerance, not bitwise.
    fn assert_vec_close_f64(actual: &[f64], reference: &[f64], tol: f64) {
        let err = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err <= tol * norm.max(1.0),
            "err={err} norm={norm} tol={tol}"
        );
    }

    #[test]
    fn test_matvec_parity_f64() {
        for n in [3, 64, 512] {
            let m = DenseMatrix::from_fn(n, n, |i, j| prng(i * n + j));
            let x: Vec<f64> = (0..n).map(|k| prng(n * n + k)).collect();
            let y = m.matvec(&x);
            let y_ref: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| m[[i, j]] * x[j]).sum())
                .collect();
            assert_vec_close_f64(&y, &y_ref, 1e-12);

            let mut yt = vec![0.0; n];
            m.matvec_transpose_into(&x, &mut yt);
            let yt_ref: Vec<f64> = (0..n)
                .map(|j| (0..n).map(|i| m[[i, j]] * x[i]).sum())
                .collect();
            assert_vec_close_f64(&yt, &yt_ref, 1e-12);
        }
    }

    #[test]
    fn test_matvec_parity_c64() {
        use num_complex::Complex64;
        for n in [3, 64, 512] {
            let m = DenseMatrix::from_fn(n, n, |i, j| {
                Complex64::new(prng(i * n + j), prng(n * n + i * n + j))
            });
            let x: Vec<Complex64> = (0..n)
                .map(|k| Complex64::new(prng(2 * n * n + k), prng(3 * n * n + k)))
                .collect();

            let y = m.matvec(&x);
            let y_ref: Vec<Complex64> = (0..n)
                .map(|i| (0..n).map(|j| m[[i, j]] * x[j]).sum())
                .collect();
            let err = y
                .iter()
                .zip(&y_ref)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let norm = y_ref.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(err <= 1e-12 * norm.max(1.0), "err={err} norm={norm}");

            let mut yh = vec![Complex64::new(0.0, 0.0); n];
            m.matvec_hermitian_into(&x, &mut yh);
            let yh_ref: Vec<Complex64> = (0..n)
                .map(|j| (0..n).map(|i| m[[i, j]].conj() * x[i]).sum())
                .collect();
            let err = yh
                .iter()
                .zip(&yh_ref)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let norm = yh_ref.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(err <= 1e-12 * norm.max(1.0), "err={err} norm={norm}");
        }
    }

    #[test]
    fn adjoint_conjugates_and_transposes() {
        use num_complex::Complex64;
        let a = DenseMatrix::from_row_major_vec(
            2,
            3,
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(-1.0, 2.0),
                Complex64::new(3.0, 3.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let h = a.adjoint();
        assert_eq!(h.shape(), (3, 2));
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(h[[j, i]], a[[i, j]].conj());
            }
        }
        // Adjoint is an involution: (Aᴴ)ᴴ = A.
        let round_trip = h.adjoint();
        assert_eq!(round_trip.shape(), (2, 3));
        assert_eq!(round_trip.as_slice(), a.as_slice());
    }

    #[test]
    fn adjoint_matches_transpose_on_real() {
        let a = DenseMatrix::from_fn(4, 3, |i, j| prng(i * 3 + j));
        assert_eq!(a.adjoint().as_slice(), a.transposed().as_slice());
    }
}
