//! Dense row-major N-dimensional arrays and the complex [`DiffTensor`].
//!
//! This module replaces `ndarray` for `autodiff` with a deliberately small
//! dense tensor covering exactly the operations the modules use: dynamic-rank
//! storage, strided sub-views ([`AxisView`]/[`AxisViewMut`]) with write-through
//! semantics, contiguous 2-D views ([`Mat`]/[`MatMut`]), owned reshapes, and
//! elementwise maps. Everything is C-contiguous by construction, so the
//! strided fallback paths of the original code keep working unchanged.

use num_complex::Complex;
use std::ops::{Index, IndexMut};

/// Axis selector, mirroring `ndarray::Axis` at the call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis(pub usize);

/// Dynamic-rank shape argument, mirroring `ndarray::IxDyn` at the call sites.
#[derive(Debug, Clone, Copy)]
pub struct IxDyn<'a>(pub &'a [usize]);

/// Dense row-major N-dimensional array.
#[derive(Debug, Clone, PartialEq)]
pub struct NdArray<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T> NdArray<T> {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the array holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Shape (extent of each axis).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of axes.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Dimensions as a pair (panics unless 2-D).
    ///
    /// # Panics
    ///
    /// Panics unless the array is 2-D.
    #[must_use]
    pub fn dim2(&self) -> (usize, usize) {
        assert_eq!(self.shape.len(), 2, "expected a 2-D array");
        (self.shape[0], self.shape[1])
    }

    /// Dimensions as a triple (panics unless 3-D).
    ///
    /// # Panics
    ///
    /// Panics unless the array is 3-D.
    #[must_use]
    pub fn dim3(&self) -> (usize, usize, usize) {
        assert_eq!(self.shape.len(), 3, "expected a 3-D array");
        (self.shape[0], self.shape[1], self.shape[2])
    }

    /// Dimensions as a quadruple (panics unless 4-D).
    ///
    /// # Panics
    ///
    /// Panics unless the array is 4-D.
    #[must_use]
    pub fn dim4(&self) -> (usize, usize, usize, usize) {
        assert_eq!(self.shape.len(), 4, "expected a 4-D array");
        (self.shape[0], self.shape[1], self.shape[2], self.shape[3])
    }

    /// Flat row-major element access.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[T]> {
        Some(&self.data)
    }

    /// Flat row-major mutable element access.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data)
    }

    /// Iterate over elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Mutably iterate over elements in row-major order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    fn offset(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.shape.len(), "rank mismatch");
        let mut offset = 0;
        let mut stride = 1;
        for (axis, &extent) in self.shape.iter().enumerate().rev() {
            debug_assert!(index[axis] < extent, "index out of bounds");
            offset += index[axis] * stride;
            stride *= extent;
        }
        offset
    }

    /// Element at a multi-index, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        if index.len() != self.shape.len() {
            return None;
        }
        if index
            .iter()
            .zip(self.shape.iter())
            .any(|(&i, &extent)| i >= extent)
        {
            return None;
        }
        Some(&self.data[self.offset(index)])
    }

    /// Sub-view with `axis` fixed at `index` (rank minus one, write-through
    /// borrowed).
    #[must_use]
    pub fn index_axis(&self, axis: Axis, index: usize) -> AxisView<'_, T> {
        let mut shape = self.shape.clone();
        let mut strides = contiguous_strides(&shape);
        let offset = index * strides[axis.0];
        shape.remove(axis.0);
        strides.remove(axis.0);
        AxisView {
            data: &self.data,
            shape,
            strides,
            offset,
        }
    }

    /// Mutable sub-view with `axis` fixed at `index`.
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> AxisViewMut<'_, T> {
        let mut shape = self.shape.clone();
        let mut strides = contiguous_strides(&shape);
        let offset = index * strides[axis.0];
        shape.remove(axis.0);
        strides.remove(axis.0);
        AxisViewMut {
            data: &mut self.data,
            shape,
            strides,
            offset,
        }
    }

    /// Owned copy with a new shape (same element order, total size must match).
    ///
    /// # Errors
    ///
    /// Returns an error when the new shape holds a different element count.
    pub fn reshaped(&self, shape: &[usize]) -> Result<Self, String>
    where
        T: Clone,
    {
        let len: usize = shape.iter().product();
        if len != self.data.len() {
            return Err(format!(
                "reshape: shape {shape:?} ({len} elements) does not match {} elements",
                self.data.len()
            ));
        }
        Ok(Self {
            shape: shape.to_vec(),
            data: self.data.clone(),
        })
    }

    /// Owned transpose of a 2-D array.
    ///
    /// # Panics
    ///
    /// Panics unless the array is 2-D.
    #[must_use]
    pub fn transposed(&self) -> Self
    where
        T: Clone,
    {
        assert_eq!(self.shape.len(), 2, "transposed needs a 2-D array");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(self.data.len());
        for j in 0..cols {
            for i in 0..rows {
                data.push(self.data[i * cols + j].clone());
            }
        }
        Self {
            shape: vec![cols, rows],
            data,
        }
    }

    /// Elementwise map producing a new array (possibly of a new element type).
    pub fn mapv<U, F>(&self, mut f: F) -> NdArray<U>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        NdArray {
            shape: self.shape.clone(),
            data: self.data.iter().cloned().map(&mut f).collect(),
        }
    }

    /// Whole-tensor immutable view (`ndarray::Array::view` equivalent).
    #[inline]
    #[must_use]
    pub fn view(&self) -> &Self {
        self
    }

    /// Whole-tensor mutable view.
    #[inline]
    pub fn view_mut(&mut self) -> &mut Self {
        self
    }

    /// Raw pointer to the element buffer start.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Mutable raw pointer to the element buffer start.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Row-major (C-order) strides in elements, one per axis.
    #[must_use]
    pub fn strides(&self) -> Vec<isize> {
        let mut out = vec![0isize; self.shape.len()];
        let mut acc: isize = 1;
        for (slot, &dim) in out.iter_mut().zip(self.shape.iter()).rev() {
            *slot = acc;
            acc *= dim.cast_signed();
        }
        out
    }

    /// Contiguous 2-D view (panics unless the array is 2-D).
    ///
    /// # Panics
    ///
    /// Panics unless the array is 2-D.
    #[must_use]
    pub fn as_mat(&self) -> Mat<'_, T> {
        assert_eq!(self.shape.len(), 2, "expected a 2-D array");
        Mat {
            data: &self.data,
            rows: self.shape[0],
            cols: self.shape[1],
        }
    }

    /// Contiguous mutable 2-D view (panics unless the array is 2-D).
    ///
    /// # Panics
    ///
    /// Panics unless the array is 2-D.
    pub fn as_mat_mut(&mut self) -> MatMut<'_, T> {
        assert_eq!(self.shape.len(), 2, "expected a 2-D array");
        MatMut {
            data: &mut self.data,
            rows: self.shape[0],
            cols: self.shape[1],
        }
    }
}

impl<T: Clone> NdArray<T> {
    /// Array filled with `value`.
    pub fn from_elem(shape: impl IntoShape, value: T) -> Self {
        let extents = shape.into_shape_vec();
        let len: usize = extents.iter().product();
        Self {
            shape: extents,
            data: vec![value; len],
        }
    }

    /// Zero-filled array.
    pub fn zeros(shape: impl IntoShape) -> Self
    where
        T: num_traits::Zero,
    {
        Self::from_elem(shape, T::zero())
    }

    /// One-filled array.
    pub fn ones(shape: impl IntoShape) -> Self
    where
        T: num_traits::One,
    {
        Self::from_elem(shape, T::one())
    }

    /// Array from a shape and row-major data (length must match).
    ///
    /// # Errors
    ///
    /// Returns an error when the data length does not match the shape.
    pub fn from_shape_vec(shape: impl IntoShape, data: Vec<T>) -> Result<Self, String> {
        let extents = shape.into_shape_vec();
        let len: usize = extents.iter().product();
        if data.len() != len {
            return Err(format!(
                "from_shape_vec: shape {extents:?} ({len} elements) does not match {} values",
                data.len()
            ));
        }
        Ok(Self {
            shape: extents,
            data,
        })
    }

    /// Fill with `value`.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

/// Accepted shape argument for constructors and reshapes.
///
/// Mirrors the `ndarray` call sites this shim replaces: rank tuples such as
/// `(rows, cols)`, fixed arrays, slices, [`IxDyn`], and owned vectors.
pub trait IntoShape {
    /// Consume the argument into an owned extent vector.
    fn into_shape_vec(self) -> Vec<usize>;
}

impl IntoShape for IxDyn<'_> {
    fn into_shape_vec(self) -> Vec<usize> {
        self.0.to_vec()
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape_vec(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

macro_rules! impl_shape_tuple {
    ($n:literal, ($($v:ident : $t:ty),+)) => {
        impl IntoShape for ($($t,)+) {
            fn into_shape_vec(self) -> Vec<usize> {
                #[allow(non_snake_case)]
                let ($($v,)+) = self;
                let _ = $n;
                vec![$($v),+]
            }
        }
    };
}

impl_shape_tuple!(1, (A: usize));
impl_shape_tuple!(2, (A: usize, B: usize));
impl_shape_tuple!(3, (A: usize, B: usize, C: usize));
impl_shape_tuple!(4, (A: usize, B: usize, C: usize, D: usize));
impl_shape_tuple!(5, (A: usize, B: usize, C: usize, D: usize, E: usize));
impl_shape_tuple!(6, (A: usize, B: usize, C: usize, D: usize, E: usize, F: usize));

/// Shape argument that also names its element-index type for [`NdArray::from_shape_fn`].
pub trait ShapeWithIndex: IntoShape {
    /// Index value passed to the element closure (a rank tuple, or a
    /// coordinate vector for dynamic shapes).
    type Index;
    /// Decode a row-major linear offset into an index value.
    fn index_of(shape: &[usize], linear: usize) -> Self::Index;
}

impl ShapeWithIndex for (usize,) {
    type Index = (usize,);
    fn index_of(_shape: &[usize], linear: usize) -> Self::Index {
        (linear,)
    }
}

impl ShapeWithIndex for (usize, usize) {
    type Index = (usize, usize);
    fn index_of(shape: &[usize], linear: usize) -> Self::Index {
        (linear / shape[1], linear % shape[1])
    }
}

impl ShapeWithIndex for (usize, usize, usize) {
    type Index = (usize, usize, usize);
    fn index_of(shape: &[usize], linear: usize) -> Self::Index {
        let row = linear / shape[2];
        (row / shape[1], row % shape[1], linear % shape[2])
    }
}

impl ShapeWithIndex for (usize, usize, usize, usize) {
    type Index = (usize, usize, usize, usize);
    fn index_of(shape: &[usize], linear: usize) -> Self::Index {
        let inner: usize = shape[1..].iter().product();
        let rest = linear % inner;
        let mid = rest / shape[3];
        (
            linear / inner,
            mid / shape[2],
            mid % shape[2],
            rest % shape[3],
        )
    }
}

impl ShapeWithIndex for IxDyn<'_> {
    type Index = Vec<usize>;
    fn index_of(shape: &[usize], linear: usize) -> Self::Index {
        decode_index(shape, linear)
    }
}

impl ShapeWithIndex for Vec<usize> {
    type Index = Vec<usize>;
    fn index_of(shape: &[usize], linear: usize) -> Self::Index {
        decode_index(shape, linear)
    }
}

impl ShapeWithIndex for &[usize] {
    type Index = Vec<usize>;
    fn index_of(shape: &[usize], linear: usize) -> Self::Index {
        decode_index(shape, linear)
    }
}

fn decode_index(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    // Callers only decode offsets below the element count, so every extent
    // on the path is nonzero; a zero extent panics via division by zero.
    for (slot, &extent) in index.iter_mut().zip(shape.iter()).rev() {
        *slot = linear % extent;
        linear /= extent;
    }
    index
}

impl<T> NdArray<T> {
    /// Array built elementwise from a shape and an index closure.
    pub fn from_shape_fn<S: ShapeWithIndex>(shape: S, mut f: impl FnMut(S::Index) -> T) -> Self {
        let extents = shape.into_shape_vec();
        let len: usize = extents.iter().product();
        let mut data = Vec::with_capacity(len);
        for linear in 0..len {
            data.push(f(S::index_of(&extents, linear)));
        }
        Self {
            shape: extents,
            data,
        }
    }

    /// 1-D array from a vector.
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        Self::from_iter(data)
    }

    /// Identity conversion (`ndarray::Array::into_dyn` equivalent; the
    /// shim is always dynamically ranked).
    #[must_use]
    pub fn into_dyn(self) -> Self {
        self
    }

    /// Owned reshape with an explicit shape argument.
    ///
    /// # Errors
    ///
    /// Returns an error when the new shape holds a different element count.
    pub fn into_shape_with_order(&self, shape: impl IntoShape) -> Result<Self, String>
    where
        T: Clone,
    {
        let extents = shape.into_shape_vec();
        self.reshaped(&extents)
    }

    /// Owned reshape (alias kept for ported call sites).
    ///
    /// # Errors
    ///
    /// Returns an error when the new shape holds a different element count.
    pub fn into_shape(&self, shape: impl IntoShape) -> Result<Self, String>
    where
        T: Clone,
    {
        self.into_shape_with_order(shape)
    }

    /// Owned transpose for 2-D arrays; a clone for 1-D.
    ///
    /// # Panics
    ///
    /// Panics unless the array is 1-D or 2-D.
    #[must_use]
    pub fn t(&self) -> Self
    where
        T: Clone,
    {
        if self.shape.len() == 1 {
            return self.clone();
        }
        self.transposed()
    }

    /// Owned axis permutation.
    ///
    /// # Panics
    ///
    /// Panics on rank mismatch, out-of-range axes, or duplicate axes.
    #[must_use]
    pub fn permuted_axes(&self, axes: impl AsRef<[usize]>) -> Self
    where
        T: Clone,
    {
        let axes = axes.as_ref();
        assert_eq!(axes.len(), self.shape.len(), "permuted_axes: rank mismatch");
        let mut seen = vec![false; axes.len()];
        for &axis in axes {
            assert!(axis < axes.len(), "permuted_axes: axis out of range");
            assert!(!seen[axis], "permuted_axes: duplicate axis");
            seen[axis] = true;
        }
        let new_shape: Vec<usize> = axes.iter().map(|&axis| self.shape[axis]).collect();
        let strides = contiguous_strides(&self.shape);
        let len = self.data.len();
        let mut data = Vec::with_capacity(len);
        for linear in 0..len {
            let new_index = decode_index(&new_shape, linear);
            let mut old_index = vec![0; axes.len()];
            for (new_axis, &old_axis) in axes.iter().enumerate() {
                old_index[old_axis] = new_index[new_axis];
            }
            let offset: usize = old_index
                .iter()
                .zip(strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            data.push(self.data[offset].clone());
        }
        Self {
            shape: new_shape,
            data,
        }
    }

    /// Matrix product (`2-D x 2-D -> 2-D`, `2-D x 1-D -> 1-D`).
    ///
    /// # Panics
    ///
    /// Panics on incompatible inner dimensions or unsupported ranks.
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> Self
    where
        T: Clone + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        match (self.shape.as_slice(), rhs.shape.as_slice()) {
            ([m, k1], [k2, n]) => {
                assert_eq!(k1, k2, "dot: incompatible inner dimensions");
                let (m, k, n) = (*m, *k1, *n);
                let mut out = Vec::with_capacity(m * n);
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = T::zero();
                        for p in 0..k {
                            acc = acc + self.data[i * k + p].clone() * rhs.data[p * n + j].clone();
                        }
                        out.push(acc);
                    }
                }
                Self {
                    shape: vec![m, n],
                    data: out,
                }
            }
            ([m, k1], [k2]) => {
                assert_eq!(k1, k2, "dot: incompatible inner dimensions");
                let (m, k) = (*m, *k1);
                let mut out = Vec::with_capacity(m);
                for i in 0..m {
                    let mut acc = T::zero();
                    for p in 0..k {
                        acc = acc + self.data[i * k + p].clone() * rhs.data[p].clone();
                    }
                    out.push(acc);
                }
                Self {
                    shape: vec![m],
                    data: out,
                }
            }
            _ => panic!("dot: unsupported shapes {:?} x {:?}", self.shape, rhs.shape),
        }
    }

    /// Elementwise map over shared references.
    pub fn map<F, U>(&self, f: F) -> NdArray<U>
    where
        F: FnMut(&T) -> U,
    {
        NdArray {
            shape: self.shape.clone(),
            data: self.data.iter().map(f).collect(),
        }
    }

    /// Largest element, or `None` when empty.
    #[must_use]
    pub fn max(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Smallest element, or `None` when empty.
    #[must_use]
    pub fn min(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Flat mutable element access (always contiguous, hence always `Some`).
    pub fn as_slice_memory_order_mut(&mut self) -> Option<&mut [T]> {
        Some(&mut self.data)
    }

    /// Always `true`: shim storage is dense row-major by construction.
    #[must_use]
    pub fn is_standard_layout(&self) -> bool {
        let _ = self;
        true
    }

    /// Identity borrow: storage is already in standard layout.
    #[must_use]
    pub fn as_standard_layout(&self) -> &Self {
        self
    }

    /// Identity conversion for ported view call sites.
    #[must_use]
    pub fn into_owned(self) -> Self {
        self
    }
}

impl IntoShape for usize {
    fn into_shape_vec(self) -> Vec<usize> {
        vec![self]
    }
}

impl<T> From<Vec<T>> for NdArray<T> {
    fn from(data: Vec<T>) -> Self {
        Self::from_vec(data)
    }
}

impl<T> std::iter::FromIterator<T> for NdArray<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        let len = data.len();
        Self {
            shape: vec![len],
            data,
        }
    }
}

impl<T> std::ops::Index<usize> for NdArray<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        assert_eq!(self.shape.len(), 1, "expected a 1-D array");
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for NdArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert_eq!(self.shape.len(), 1, "expected a 1-D array");
        &mut self.data[index]
    }
}

impl<T> std::ops::Sub<&NdArray<T>> for &NdArray<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    type Output = NdArray<T>;
    fn sub(self, rhs: &NdArray<T>) -> NdArray<T> {
        assert_eq!(self.shape, rhs.shape, "Sub: shape mismatch");
        NdArray {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a.clone() - b.clone())
                .collect(),
        }
    }
}

impl<T> std::ops::Mul<T> for NdArray<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            shape: self.shape,
            data: self.data.into_iter().map(|v| v * rhs.clone()).collect(),
        }
    }
}

impl<T> std::ops::Mul<T> for &NdArray<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    type Output = NdArray<T>;
    fn mul(self, rhs: T) -> NdArray<T> {
        NdArray {
            shape: self.shape.clone(),
            data: self.data.iter().map(|v| v.clone() * rhs.clone()).collect(),
        }
    }
}

impl<T> std::ops::Div<T> for NdArray<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Self {
            shape: self.shape,
            data: self.data.into_iter().map(|v| v / rhs.clone()).collect(),
        }
    }
}

impl<T> std::ops::Div<T> for &NdArray<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    type Output = NdArray<T>;
    fn div(self, rhs: T) -> NdArray<T> {
        NdArray {
            shape: self.shape.clone(),
            data: self.data.iter().map(|v| v.clone() / rhs.clone()).collect(),
        }
    }
}

impl<T> std::ops::AddAssign<&NdArray<T>> for NdArray<T>
where
    T: Clone + std::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: &NdArray<T>) {
        assert_eq!(self.shape, rhs.shape, "AddAssign: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b.clone();
        }
    }
}

macro_rules! impl_complex_scalar_ops {
    ($real:ty) => {
        impl std::ops::Mul<$real> for NdArray<num_complex::Complex<$real>> {
            type Output = Self;
            fn mul(self, rhs: $real) -> Self {
                Self {
                    shape: self.shape,
                    data: self
                        .data
                        .into_iter()
                        .map({
                            let scalar = rhs;
                            move |v| v * scalar
                        })
                        .collect(),
                }
            }
        }

        impl std::ops::Mul<$real> for &NdArray<num_complex::Complex<$real>> {
            type Output = NdArray<num_complex::Complex<$real>>;
            fn mul(self, rhs: $real) -> NdArray<num_complex::Complex<$real>> {
                NdArray {
                    shape: self.shape.clone(),
                    data: self
                        .data
                        .iter()
                        .map({
                            let scalar = rhs;
                            move |v| *v * scalar
                        })
                        .collect(),
                }
            }
        }

        impl std::ops::Div<$real> for NdArray<num_complex::Complex<$real>> {
            type Output = Self;
            fn div(self, rhs: $real) -> Self {
                Self {
                    shape: self.shape,
                    data: self
                        .data
                        .into_iter()
                        .map({
                            let scalar = rhs;
                            move |v| v / scalar
                        })
                        .collect(),
                }
            }
        }

        impl std::ops::Div<$real> for &NdArray<num_complex::Complex<$real>> {
            type Output = NdArray<num_complex::Complex<$real>>;
            fn div(self, rhs: $real) -> NdArray<num_complex::Complex<$real>> {
                NdArray {
                    shape: self.shape.clone(),
                    data: self
                        .data
                        .iter()
                        .map({
                            let scalar = rhs;
                            move |v| *v / scalar
                        })
                        .collect(),
                }
            }
        }
    };
}

impl_complex_scalar_ops!(f32);
impl_complex_scalar_ops!(f64);

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

macro_rules! impl_nd_index {
    ($n:expr) => {
        impl<T> Index<[usize; $n]> for NdArray<T> {
            type Output = T;
            fn index(&self, index: [usize; $n]) -> &T {
                &self.data[self.offset(&index)]
            }
        }
        impl<T> IndexMut<[usize; $n]> for NdArray<T> {
            fn index_mut(&mut self, index: [usize; $n]) -> &mut T {
                let offset = self.offset(&index);
                &mut self.data[offset]
            }
        }
    };
}

impl_nd_index!(1);
impl_nd_index!(2);
impl_nd_index!(3);
impl_nd_index!(4);
impl_nd_index!(5);

impl<'a, T> IntoIterator for &'a NdArray<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut NdArray<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> std::ops::Add<&NdArray<T>> for &NdArray<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    type Output = NdArray<T>;
    fn add(self, rhs: &NdArray<T>) -> NdArray<T> {
        assert_eq!(self.shape(), rhs.shape(), "elementwise add shape mismatch");
        NdArray {
            shape: self.shape.clone(),
            data: self
                .iter()
                .zip(rhs.iter())
                .map(|(a, b)| a.clone() + b.clone())
                .collect(),
        }
    }
}

/// Borrowed strided sub-view of an [`NdArray`], in logical (row-major) order.
#[derive(Debug)]
pub struct AxisView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<'a, T> AxisView<'a, T> {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the view holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shape (extent of each axis).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Iterate over elements in logical row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let offset = self.offset;
        let total: usize = shape.iter().product();
        (0..total).map(move |linear| {
            let mut rest = linear;
            let mut index = offset;
            for axis in (0..shape.len()).rev() {
                let extent = shape[axis];
                let coord = if extent == 0 { 0 } else { rest % extent };
                rest /= extent.max(1);
                index += coord * strides[axis];
            }
            &self.data[index]
        })
    }

    /// Element at a multi-index.
    #[must_use]
    pub fn get(&self, index: &[usize]) -> Option<&'a T> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut flat = self.offset;
        for ((&i, &extent), &stride) in index.iter().zip(self.shape.iter()).zip(self.strides.iter())
        {
            if i >= extent {
                return None;
            }
            flat += i * stride;
        }
        Some(&self.data[flat])
    }

    /// Nested sub-view with `axis` fixed at `index`.
    #[must_use]
    pub fn index_axis(&self, axis: Axis, index: usize) -> AxisView<'_, T> {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let offset = self.offset + index * strides[axis.0];
        shape.remove(axis.0);
        strides.remove(axis.0);
        AxisView {
            data: self.data,
            shape,
            strides,
            offset,
        }
    }

    /// Owned dense copy in logical order.
    #[must_use]
    pub fn to_owned(&self) -> NdArray<T>
    where
        T: Clone,
    {
        NdArray {
            shape: self.shape.clone(),
            data: self.iter().cloned().collect(),
        }
    }

    /// Elementwise map producing a new dense array.
    pub fn mapv<U, F>(&self, f: F) -> NdArray<U>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        NdArray {
            shape: self.shape.clone(),
            data: self.iter().cloned().map(f).collect(),
        }
    }
}

/// Mutable borrowed strided sub-view with write-through semantics.
#[derive(Debug)]
pub struct AxisViewMut<'a, T> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<T> AxisViewMut<'_, T> {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the view holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shape (extent of each axis).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn flat_indices(&self) -> Vec<usize> {
        let total: usize = self.shape.iter().product();
        let mut indices = Vec::with_capacity(total);
        for linear in 0..total {
            let mut rest = linear;
            let mut index = self.offset;
            for axis in (0..self.shape.len()).rev() {
                let extent = self.shape[axis];
                let coord = if extent == 0 { 0 } else { rest % extent };
                rest /= extent.max(1);
                index += coord * self.strides[axis];
            }
            indices.push(index);
        }
        indices
    }

    /// Iterate over elements in logical row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.flat_indices()
            .into_iter()
            .map(|index| &self.data[index])
    }

    /// Mutably iterate over elements in logical row-major order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        let indices = self.flat_indices();
        let data = &mut *self.data;
        indices.into_iter().map(|index| {
            // SAFETY: each logical element maps to a distinct flat index.
            unsafe { &mut *(data.as_mut_ptr().add(index)) }
        })
    }

    /// Nested sub-view with `axis` fixed at `index`.
    #[must_use]
    pub fn index_axis(&self, axis: Axis, index: usize) -> AxisView<'_, T> {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let offset = self.offset + index * strides[axis.0];
        shape.remove(axis.0);
        strides.remove(axis.0);
        AxisView {
            data: self.data,
            shape,
            strides,
            offset,
        }
    }

    /// Nested mutable sub-view with `axis` fixed at `index`.
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> AxisViewMut<'_, T> {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let offset = self.offset + index * strides[axis.0];
        shape.remove(axis.0);
        strides.remove(axis.0);
        AxisViewMut {
            data: &mut *self.data,
            shape,
            strides,
            offset,
        }
    }

    /// Elementwise in-place map.
    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
        T: Clone,
    {
        for value in self.iter_mut() {
            let owned = value.clone();
            *value = f(owned);
        }
    }

    /// Owned dense copy in logical order.
    #[must_use]
    pub fn to_owned(&self) -> NdArray<T>
    where
        T: Clone,
    {
        NdArray {
            shape: self.shape.clone(),
            data: self.iter().cloned().collect(),
        }
    }
}

/// Contiguous borrowed 2-D view.
#[derive(Debug)]
pub struct Mat<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> Mat<'a, T> {
    /// Dimensions as `(rows, cols)`.
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Iterate over elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'a, T> {
        self.data.iter()
    }

    /// Owned dense copy.
    #[must_use]
    pub fn to_owned(&self) -> NdArray<T>
    where
        T: Clone,
    {
        NdArray {
            shape: vec![self.rows, self.cols],
            data: self.data.to_vec(),
        }
    }

    /// Owned transpose.
    #[must_use]
    pub fn transposed(&self) -> NdArray<T>
    where
        T: Clone,
    {
        self.to_owned().transposed()
    }

    /// Elementwise map producing a new dense array.
    pub fn mapv<U, F>(&self, f: F) -> NdArray<U>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        self.to_owned().mapv(f)
    }
}

impl<T> Index<[usize; 2]> for Mat<'_, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &T {
        &self.data[index[0] * self.cols + index[1]]
    }
}

impl<'a, T> IntoIterator for &'a Mat<'_, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

/// Contiguous mutable borrowed 2-D view.
#[derive(Debug)]
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<T> MatMut<'_, T> {
    /// Dimensions as `(rows, cols)`.
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Iterate over elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Mutably iterate over elements in row-major order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T> Index<[usize; 2]> for MatMut<'_, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &T {
        &self.data[index[0] * self.cols + index[1]]
    }
}

impl<'a, T> IntoIterator for &'a MatMut<'_, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut MatMut<'_, T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> std::slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> IndexMut<[usize; 2]> for MatMut<'_, T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        &mut self.data[index[0] * self.cols + index[1]]
    }
}

/// Complex-valued tensor used for frequency-domain signals and responses.
///
/// Supports dynamic rank so that `(B, M, N)`, `(M, N, N)`, and `(M, N)`
/// can be represented uniformly.
#[derive(Debug, Clone)]
pub struct DiffTensor<T = f64> {
    /// Row-major dense element storage.
    pub data: NdArray<Complex<T>>,
}

impl<T: num_traits::Float> DiffTensor<T> {
    /// Create a tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: IxDyn<'_>) -> Self {
        Self {
            data: NdArray::zeros(shape),
        }
    }

    /// Create a tensor from an existing dense array.
    #[must_use]
    pub fn from_array(data: NdArray<Complex<T>>) -> Self {
        Self { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zeros_creates_correct_shape_and_zero_filled() {
        let t = DiffTensor::<f64>::zeros(IxDyn(&[2, 3]));
        assert_eq!(t.data.shape(), &[2, 3]);
        for c in &t.data {
            assert_abs_diff_eq!(c.re, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn from_array_preserves_shape_and_values() {
        let values = vec![Complex::new(1.0, -1.0); 6];
        let arr = NdArray::from_shape_vec(IxDyn(&[2, 3]), values).unwrap();
        let t = DiffTensor::<f64>::from_array(arr.clone());
        assert_eq!(t.data.shape(), &[2, 3]);
        for (actual, expected) in t.data.iter().zip(arr.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn index_axis_mut_writes_through() {
        let mut a = NdArray::from_shape_vec(IxDyn(&[2, 3]), vec![0.0; 6]).unwrap();
        {
            let mut view = a.index_axis_mut(Axis(0), 1);
            for x in view.iter_mut() {
                *x = 5.0;
            }
        }
        assert_eq!(a.as_slice().unwrap(), &[0.0, 0.0, 0.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn nested_index_axis_mut_writes_through() {
        let mut a =
            NdArray::from_shape_vec(IxDyn(&[2, 2, 2]), (0..8).map(f64::from).collect()).unwrap();
        {
            let mut outer = a.index_axis_mut(Axis(1), 1);
            let mut inner = outer.index_axis_mut(Axis(1), 0);
            for x in inner.iter_mut() {
                *x *= 10.0;
            }
        }
        // Elements [0,1,0] and [1,1,0] (flat 2 and 6) scaled by 10.
        assert_eq!(
            a.as_slice().unwrap(),
            &[0.0, 1.0, 20.0, 3.0, 4.0, 5.0, 60.0, 7.0]
        );
    }

    #[test]
    fn reshape_and_transpose_match_row_major_order() {
        let a = NdArray::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(f64::from).collect()).unwrap();
        let r = a.reshaped(&[3, 2]).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = a.transposed();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.as_slice().unwrap(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
