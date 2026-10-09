//! Device-resident block operators over compute backends.
//!
//! The block assembly stays backend-neutral: each block exposes a device-buffer
//! apply, and this operator handles composition. Checked one-dimensional
//! sub-vector extraction uses the backend's view-copy contract, so CPU, CUDA,
//! and Metal can keep block inputs device-resident; output accumulation remains
//! a checked host reduction until a view-aware AXPY operation is available.

use std::ops::Range;

use crate::{Backend, BufferView, DeviceBufferLike, DeviceCsr, DeviceOperator, DeviceScalar};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLayout {
    row_sizes: Vec<usize>,
    col_sizes: Vec<usize>,
    row_offsets: Vec<usize>,
    col_offsets: Vec<usize>,
}

impl BlockLayout {
    /// Compatibility constructor. Prefer [`Self::try_new`] for user input;
    /// this historical API preserves its panic-on-invalid-input behavior.
    pub fn new(row_sizes: Vec<usize>, col_sizes: Vec<usize>) -> Self {
        assert!(
            !row_sizes.is_empty(),
            "block layout needs at least one row block"
        );
        assert!(
            !col_sizes.is_empty(),
            "block layout needs at least one column block"
        );
        assert!(
            row_sizes.iter().all(|&size| size > 0),
            "row block sizes must be non-zero"
        );
        assert!(
            col_sizes.iter().all(|&size| size > 0),
            "column block sizes must be non-zero"
        );

        let row_offsets = offsets(&row_sizes);
        let col_offsets = offsets(&col_sizes);
        Self {
            row_sizes,
            col_sizes,
            row_offsets,
            col_offsets,
        }
    }

    /// Checked block-layout constructor for user-controlled block sizes.
    pub fn try_new(row_sizes: Vec<usize>, col_sizes: Vec<usize>) -> crate::BackendResult<Self> {
        if row_sizes.is_empty() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                reason: "at least one row block is required".to_string(),
            });
        }
        if col_sizes.is_empty() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                reason: "at least one column block is required".to_string(),
            });
        }
        if row_sizes.contains(&0) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                reason: "row block sizes must be non-zero".to_string(),
            });
        }
        if col_sizes.contains(&0) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                reason: "column block sizes must be non-zero".to_string(),
            });
        }
        let row_offsets = checked_offsets(&row_sizes, "row")?;
        let col_offsets = checked_offsets(&col_sizes, "column")?;
        Ok(Self {
            row_sizes,
            col_sizes,
            row_offsets,
            col_offsets,
        })
    }

    pub fn num_block_rows(&self) -> usize {
        self.row_sizes.len()
    }

    pub fn num_block_cols(&self) -> usize {
        self.col_sizes.len()
    }

    pub fn num_rows(&self) -> usize {
        self.row_offsets.last().copied().unwrap_or(0)
    }

    pub fn num_cols(&self) -> usize {
        self.col_offsets.last().copied().unwrap_or(0)
    }

    pub fn try_row_size(&self, block_row: usize) -> crate::BackendResult<usize> {
        self.row_sizes
            .get(block_row)
            .copied()
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_row",
                reason: format!("row block {block_row} is out of range"),
            })
    }

    pub fn try_col_size(&self, block_col: usize) -> crate::BackendResult<usize> {
        self.col_sizes
            .get(block_col)
            .copied()
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_column",
                reason: format!("column block {block_col} is out of range"),
            })
    }

    pub fn try_row_range(&self, block_row: usize) -> crate::BackendResult<Range<usize>> {
        let end = block_row
            .checked_add(1)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_row",
                reason: format!("row block {block_row} is out of range"),
            })?;
        self.row_offsets
            .get(block_row..=end)
            .map(|offsets| offsets[0]..offsets[1])
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_row",
                reason: format!("row block {block_row} is out of range"),
            })
    }

    pub fn try_col_range(&self, block_col: usize) -> crate::BackendResult<Range<usize>> {
        let end = block_col
            .checked_add(1)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_column",
                reason: format!("column block {block_col} is out of range"),
            })?;
        self.col_offsets
            .get(block_col..=end)
            .map(|offsets| offsets[0]..offsets[1])
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_column",
                reason: format!("column block {block_col} is out of range"),
            })
    }

    pub fn row_size(&self, block_row: usize) -> usize {
        self.row_sizes[block_row]
    }

    pub fn col_size(&self, block_col: usize) -> usize {
        self.col_sizes[block_col]
    }

    pub fn row_range(&self, block_row: usize) -> Range<usize> {
        self.row_offsets[block_row]..self.row_offsets[block_row + 1]
    }

    pub fn col_range(&self, block_col: usize) -> Range<usize> {
        self.col_offsets[block_col]..self.col_offsets[block_col + 1]
    }
}

fn offsets(sizes: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(sizes.len() + 1);
    offsets.push(0);
    let mut total: usize = 0;
    for &size in sizes {
        total += size;
        offsets.push(total);
    }
    offsets
}

fn reserve_block_vec<T>(
    values: &mut Vec<T>,
    required: usize,
    operation: &'static str,
) -> crate::BackendResult<()> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|error| crate::BackendError::Allocation {
                reason: format!("{operation}: {error}"),
            })?;
    }
    Ok(())
}

fn zeroed_block_vec<T: Copy + Default>(
    len: usize,
    operation: &'static str,
) -> crate::BackendResult<Vec<T>> {
    crate::backend::try_zeroed_host::<T>(len, operation)
}

fn checked_offsets(sizes: &[usize], axis: &'static str) -> crate::BackendResult<Vec<usize>> {
    let capacity = sizes
        .len()
        .checked_add(1)
        .ok_or(crate::BackendError::InvalidArgument {
            operation: "block_layout_new",
            reason: format!("{axis} offset count overflow"),
        })?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(capacity)
        .map_err(|error| crate::BackendError::Allocation {
            reason: format!("block_layout_new: {error}"),
        })?;
    offsets.push(0);
    let mut total: usize = 0;
    for &size in sizes {
        total = total
            .checked_add(size)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                reason: format!("{axis} block dimensions overflow usize"),
            })?;
        offsets.push(total);
    }
    Ok(offsets)
}

pub trait DeviceBlock<B: Backend, T: DeviceScalar>: Send + Sync {
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>);

    /// Fallible single-right-hand-side application with shape and workspace
    /// failures propagated to the caller. The legacy [`Self::apply`] method
    /// remains available for already-validated hot paths.
    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let rows = self.num_rows();
        let cols = self.num_cols();
        if x.len() != cols || y.len() != rows {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_apply",
                reason: format!(
                    "expected input/output length {cols}/{rows}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        crate::backend::catch_backend_panic("device_block_apply", || self.apply(backend, x, y))?;
        backend.check_pending_error()
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, x, y) {
            log::error!(
                target: "sonium_compute::block_operator",
                "device block multi-RHS application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let rows = self.num_rows();
        let cols = self.num_cols();
        let x_len = cols
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_block_apply_colmajor",
                reason: format!("input size overflow: cols={cols}, rhs={rhs}"),
            })?;
        let y_len = rows
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_block_apply_colmajor",
                reason: format!("output size overflow: rows={rows}, rhs={rhs}"),
            })?;
        if x.len() != x_len || y.len() != y_len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_apply_colmajor",
                reason: format!(
                    "expected input/output length {x_len}/{y_len}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        if rhs == 0 || rows == 0 {
            return Ok(());
        }
        if rhs == 1 {
            return self.try_apply(backend, x, y);
        }

        let mut x_host = zeroed_block_vec::<T>(x_len, "device_block_apply_colmajor input")?;
        let mut y_host = zeroed_block_vec::<T>(y_len, "device_block_apply_colmajor output")?;
        backend.try_copy_d2h(x, &mut x_host)?;

        let mut x_col = backend
            .raw_alloc::<T>(cols)
            .map_err(crate::BackendError::from)?;
        let mut y_col = backend
            .raw_alloc::<T>(rows)
            .map_err(crate::BackendError::from)?;

        for col in 0..rhs {
            let x_range = col * cols..(col + 1) * cols;
            let y_range = col * rows..(col + 1) * rows;
            backend.try_copy_h2d(&x_host[x_range], &mut x_col)?;
            self.try_apply(backend, &x_col, &mut y_col)?;
            backend.try_copy_d2h(&y_col, &mut y_host[y_range])?;
        }

        backend.try_copy_h2d(&y_host, y)
    }
}

/// Reusable host/device staging for [`DeviceBlockOperator`] applications.
///
/// Block sub-vectors are currently staged because the backend trait does not
/// expose device buffer views. A caller-owned workspace retains the host
/// vectors and per-block device buffers across repeated applications. It is
/// tied to the layout supplied at construction and must not be shared across
/// backend contexts.
/// Input sub-vectors use checked backend view copies when available; output
/// block accumulation still uses the retained host partials.
pub struct DeviceBlockWorkspace<B: Backend, T: DeviceScalar> {
    layout: BlockLayout,
    x_host: Vec<T>,
    y_host: Vec<T>,
    x_partials: Vec<Vec<T>>,
    y_partials: Vec<Vec<T>>,
    x_buffers: Vec<Option<B::Buffer<T>>>,
    y_buffers: Vec<Option<B::Buffer<T>>>,
}

impl<B: Backend, T: DeviceScalar> DeviceBlockWorkspace<B, T> {
    /// Create an empty workspace sized for `layout`.
    pub fn new(layout: &BlockLayout) -> Self {
        Self {
            layout: layout.clone(),
            x_host: Vec::new(),
            y_host: Vec::new(),
            x_partials: layout
                .col_sizes
                .iter()
                .map(|&size| Vec::with_capacity(size))
                .collect(),
            y_partials: layout
                .row_sizes
                .iter()
                .map(|&size| Vec::with_capacity(size))
                .collect(),
            x_buffers: (0..layout.num_block_cols()).map(|_| None).collect(),
            y_buffers: (0..layout.num_block_rows()).map(|_| None).collect(),
        }
    }

    /// Release retained device buffers and clear, but do not shrink, host
    /// scratch. The workspace can be reused after this call.
    pub fn clear(&mut self) {
        self.x_host.clear();
        self.y_host.clear();
        self.x_partials.iter_mut().for_each(Vec::clear);
        self.y_partials.iter_mut().for_each(Vec::clear);
        self.x_buffers.iter_mut().for_each(|buffer| *buffer = None);
        self.y_buffers.iter_mut().for_each(|buffer| *buffer = None);
    }

    /// Host element capacity retained after warm-up.
    pub fn host_capacity(&self) -> usize {
        self.x_host.capacity()
            + self.y_host.capacity()
            + self.x_partials.iter().map(Vec::capacity).sum::<usize>()
            + self.y_partials.iter().map(Vec::capacity).sum::<usize>()
    }

    /// Number of per-block device buffers retained after warm-up.
    pub fn device_buffer_count(&self) -> usize {
        self.x_buffers
            .iter()
            .filter(|buffer| buffer.is_some())
            .count()
            + self
                .y_buffers
                .iter()
                .filter(|buffer| buffer.is_some())
                .count()
    }

    fn prepare(&mut self, layout: &BlockLayout, rhs: usize) -> crate::BackendResult<()> {
        if self.layout != *layout {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_workspace_layout",
                reason: "workspace was created for a different block layout".to_string(),
            });
        }
        let x_len =
            layout
                .num_cols()
                .checked_mul(rhs)
                .ok_or(crate::BackendError::InvalidArgument {
                    operation: "device_block_workspace_prepare",
                    reason: format!("input size overflow: cols={}, rhs={rhs}", layout.num_cols()),
                })?;
        let y_len =
            layout
                .num_rows()
                .checked_mul(rhs)
                .ok_or(crate::BackendError::InvalidArgument {
                    operation: "device_block_workspace_prepare",
                    reason: format!(
                        "output size overflow: rows={}, rhs={rhs}",
                        layout.num_rows()
                    ),
                })?;
        reserve_block_vec(
            &mut self.x_host,
            x_len,
            "device_block_workspace_prepare input",
        )?;
        reserve_block_vec(
            &mut self.y_host,
            y_len,
            "device_block_workspace_prepare output",
        )?;
        self.x_host.resize(x_len, T::zero());
        self.y_host.resize(y_len, T::zero());
        for (values, &size) in self.x_partials.iter_mut().zip(&layout.col_sizes) {
            let len = size
                .checked_mul(rhs)
                .ok_or(crate::BackendError::InvalidArgument {
                    operation: "device_block_workspace_prepare",
                    reason: format!("block input size overflow: size={size}, rhs={rhs}"),
                })?;
            reserve_block_vec(values, len, "device_block_workspace_prepare block input")?;
            values.resize(len, T::zero());
        }
        for (values, &size) in self.y_partials.iter_mut().zip(&layout.row_sizes) {
            let len = size
                .checked_mul(rhs)
                .ok_or(crate::BackendError::InvalidArgument {
                    operation: "device_block_workspace_prepare",
                    reason: format!("block output size overflow: size={size}, rhs={rhs}"),
                })?;
            reserve_block_vec(values, len, "device_block_workspace_prepare block output")?;
            values.resize(len, T::zero());
        }
        Ok(())
    }
}

impl<B: Backend, T: DeviceScalar> DeviceBlock<B, T> for DeviceCsr<B, T> {
    fn num_rows(&self) -> usize {
        self.m
    }

    fn num_cols(&self) -> usize {
        self.n
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        <Self as DeviceOperator<B, T>>::apply(self, backend, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || y.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_csr_block_apply",
                reason: format!(
                    "expected input/output length {}/{}, got {}/{}",
                    self.n,
                    self.m,
                    x.len(),
                    y.len()
                ),
            });
        }
        <Self as DeviceOperator<B, T>>::try_apply(self, backend, x, y)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmm_prebaked_colmajor::<T>(&prebaked, rhs, x, y);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let input_len = self
            .n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_csr_block_apply_colmajor",
                reason: format!("input size overflow: cols={}, rhs={rhs}", self.n),
            })?;
        let output_len = self
            .m
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_csr_block_apply_colmajor",
                reason: format!("output size overflow: rows={}, rhs={rhs}", self.m),
            })?;
        if x.len() != input_len || y.len() != output_len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_csr_block_apply_colmajor",
                reason: format!(
                    "expected input/output length {input_len}/{output_len}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        let prebaked = self.try_prebake(backend)?;
        backend.try_spmm_prebaked_colmajor::<T>(&prebaked, rhs, x, y)
    }
}

pub struct SquareDeviceBlock<O> {
    op: O,
}

impl<O> SquareDeviceBlock<O> {
    pub fn new(op: O) -> Self {
        Self { op }
    }

    pub fn operator(&self) -> &O {
        &self.op
    }
}

impl<B, T, O> DeviceBlock<B, T> for SquareDeviceBlock<O>
where
    B: Backend,
    T: DeviceScalar,
    O: DeviceOperator<B, T>,
{
    fn num_rows(&self) -> usize {
        self.op.n()
    }

    fn num_cols(&self) -> usize {
        self.op.n()
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        self.op.apply(backend, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.op.try_apply(backend, x, y)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        self.op.apply_colmajor(backend, rhs, x, y);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.op.try_apply_colmajor(backend, rhs, x, y)
    }
}

pub struct DeviceBlockOperator<B: Backend, T: DeviceScalar> {
    layout: BlockLayout,
    blocks: Vec<Option<Box<dyn DeviceBlock<B, T>>>>,
}

impl<B: Backend, T: DeviceScalar> DeviceBlockOperator<B, T> {
    /// Compatibility constructor. Prefer [`Self::try_new`] for user input;
    /// this historical API preserves its panic-on-invalid-input behavior.
    pub fn new(layout: BlockLayout) -> Self {
        assert_eq!(
            layout.num_rows(),
            layout.num_cols(),
            "device block operators exposed to Krylov solvers must be square"
        );
        let block_count = layout.num_block_rows() * layout.num_block_cols();
        Self {
            layout,
            blocks: (0..block_count).map(|_| None).collect(),
        }
    }

    /// Checked constructor for callers that cannot accept a panic on invalid
    /// global block dimensions.
    pub fn try_new(layout: BlockLayout) -> crate::BackendResult<Self> {
        if layout.num_rows() != layout.num_cols() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_new",
                reason: format!(
                    "Krylov block operators must be square, got {}x{}",
                    layout.num_rows(),
                    layout.num_cols()
                ),
            });
        }
        let block_count = layout
            .num_block_rows()
            .checked_mul(layout.num_block_cols())
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_new",
                reason: "block count overflows usize".to_string(),
            })?;
        Ok(Self {
            layout,
            blocks: (0..block_count).map(|_| None).collect(),
        })
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    /// Checked block insertion for user-controlled block coordinates and
    /// dimensions.
    pub fn try_set_block(
        &mut self,
        block_row: usize,
        block_col: usize,
        block: Box<dyn DeviceBlock<B, T>>,
    ) -> crate::BackendResult<()> {
        let index = self.checked_block_index(block_row, block_col)?;
        self.check_block_shape_checked(block_row, block_col, block.as_ref())?;
        self.blocks[index] = Some(block);
        Ok(())
    }

    pub fn try_with_block(
        mut self,
        block_row: usize,
        block_col: usize,
        block: Box<dyn DeviceBlock<B, T>>,
    ) -> crate::BackendResult<Self> {
        self.try_set_block(block_row, block_col, block)?;
        Ok(self)
    }

    pub fn set_block(
        &mut self,
        block_row: usize,
        block_col: usize,
        block: Box<dyn DeviceBlock<B, T>>,
    ) {
        self.check_block_shape(block_row, block_col, block.as_ref());
        let index = self.block_index(block_row, block_col);
        self.blocks[index] = Some(block);
    }

    pub fn with_block(
        mut self,
        block_row: usize,
        block_col: usize,
        block: Box<dyn DeviceBlock<B, T>>,
    ) -> Self {
        self.set_block(block_row, block_col, block);
        self
    }

    pub fn block(&self, block_row: usize, block_col: usize) -> Option<&dyn DeviceBlock<B, T>> {
        self.blocks[self.block_index(block_row, block_col)].as_deref()
    }

    pub fn try_block(
        &self,
        block_row: usize,
        block_col: usize,
    ) -> crate::BackendResult<Option<&dyn DeviceBlock<B, T>>> {
        let index = self.checked_block_index(block_row, block_col)?;
        Ok(self.blocks[index].as_deref())
    }

    fn checked_block_index(
        &self,
        block_row: usize,
        block_col: usize,
    ) -> crate::BackendResult<usize> {
        if block_row >= self.layout.num_block_rows() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_block",
                reason: format!("block row {block_row} is out of range"),
            });
        }
        if block_col >= self.layout.num_block_cols() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_block",
                reason: format!("block column {block_col} is out of range"),
            });
        }
        Ok(block_row * self.layout.num_block_cols() + block_col)
    }

    fn block_index(&self, block_row: usize, block_col: usize) -> usize {
        assert!(
            block_row < self.layout.num_block_rows(),
            "block row out of range"
        );
        assert!(
            block_col < self.layout.num_block_cols(),
            "block column out of range"
        );
        block_row * self.layout.num_block_cols() + block_col
    }

    fn check_block_shape(&self, block_row: usize, block_col: usize, block: &dyn DeviceBlock<B, T>) {
        let expected_rows = self.layout.row_size(block_row);
        let expected_cols = self.layout.col_size(block_col);
        assert_eq!(
            block.num_rows(),
            expected_rows,
            "device block ({block_row}, {block_col}) row count mismatch"
        );
        assert_eq!(
            block.num_cols(),
            expected_cols,
            "device block ({block_row}, {block_col}) column count mismatch"
        );
    }

    fn check_block_shape_checked(
        &self,
        block_row: usize,
        block_col: usize,
        block: &dyn DeviceBlock<B, T>,
    ) -> crate::BackendResult<()> {
        let expected_rows = self.layout.row_size(block_row);
        let expected_cols = self.layout.col_size(block_col);
        if block.num_rows() != expected_rows || block.num_cols() != expected_cols {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_set_block",
                reason: format!(
                    "block ({block_row}, {block_col}) expected {}x{}, got {}x{}",
                    expected_rows,
                    expected_cols,
                    block.num_rows(),
                    block.num_cols()
                ),
            });
        }
        Ok(())
    }
}

impl<B: Backend, T: DeviceScalar> DeviceBlockOperator<B, T> {
    /// Create a reusable staging workspace for this operator's layout.
    pub fn new_workspace(&self) -> DeviceBlockWorkspace<B, T> {
        DeviceBlockWorkspace::new(&self.layout)
    }

    /// Checked single-RHS application using caller-owned reusable staging.
    pub fn try_apply_with_workspace(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
        workspace: &mut DeviceBlockWorkspace<B, T>,
    ) -> crate::BackendResult<()> {
        self.try_apply_colmajor_with_workspace(backend, 1, x, y, workspace)
    }

    /// Checked column-major multi-RHS application using caller-owned staging.
    ///
    /// After the first call, host scratch and all buffers for present blocks
    /// are retained by `workspace`; only input/output transfers and block
    /// kernels remain in the steady-state path.
    pub fn try_apply_colmajor_with_workspace(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
        workspace: &mut DeviceBlockWorkspace<B, T>,
    ) -> crate::BackendResult<()> {
        let rows = self.layout.num_rows();
        let cols = self.layout.num_cols();
        let x_len = cols
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_apply_colmajor",
                reason: format!("input size overflow: cols={cols}, rhs={rhs}"),
            })?;
        let y_len = rows
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_apply_colmajor",
                reason: format!("output size overflow: rows={rows}, rhs={rhs}"),
            })?;
        if x.len() != x_len || y.len() != y_len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_apply_colmajor",
                reason: format!(
                    "expected input/output length {x_len}/{y_len}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        if rhs == 0 || rows == 0 {
            return Ok(());
        }

        workspace.prepare(&self.layout, rhs)?;
        workspace.y_host.fill(T::zero());

        for block_row in 0..self.layout.num_block_rows() {
            let row_range = self.layout.row_range(block_row);
            for block_col in 0..self.layout.num_block_cols() {
                let Some(block) = self.block(block_row, block_col) else {
                    continue;
                };
                let col_range = self.layout.col_range(block_col);
                let block_cols = col_range.len();
                let block_rows = row_range.len();
                let x_len =
                    block_cols
                        .checked_mul(rhs)
                        .ok_or(crate::BackendError::InvalidArgument {
                            operation: "device_block_operator_apply_colmajor",
                            reason: format!(
                                "block input size overflow: cols={block_cols}, rhs={rhs}"
                            ),
                        })?;
                let replace_x = workspace.x_buffers[block_col]
                    .as_ref()
                    .is_none_or(|buffer| buffer.len() != x_len);
                if replace_x {
                    workspace.x_buffers[block_col] = Some(
                        backend
                            .raw_alloc::<T>(x_len)
                            .map_err(crate::BackendError::from)?,
                    );
                }
                let y_len =
                    block_rows
                        .checked_mul(rhs)
                        .ok_or(crate::BackendError::InvalidArgument {
                            operation: "device_block_operator_apply_colmajor",
                            reason: format!(
                                "block output size overflow: rows={block_rows}, rhs={rhs}"
                            ),
                        })?;
                let replace_y = workspace.y_buffers[block_row]
                    .as_ref()
                    .is_none_or(|buffer| buffer.len() != y_len);
                if replace_y {
                    workspace.y_buffers[block_row] = Some(
                        backend
                            .raw_alloc::<T>(y_len)
                            .map_err(crate::BackendError::from)?,
                    );
                }

                let x_dev = workspace.x_buffers[block_col].as_mut().ok_or(
                    crate::BackendError::OperationFailed {
                        operation: "device_block_operator_apply_colmajor",
                        reason: "input workspace allocation disappeared".to_string(),
                    },
                )?;
                let y_dev = workspace.y_buffers[block_row].as_mut().ok_or(
                    crate::BackendError::OperationFailed {
                        operation: "device_block_operator_apply_colmajor",
                        reason: "output workspace allocation disappeared".to_string(),
                    },
                )?;
                let partial = &mut workspace.y_partials[block_row];
                partial.fill(T::zero());
                for col in 0..rhs {
                    let source_start = col
                        .checked_mul(cols)
                        .and_then(|offset| offset.checked_add(col_range.start))
                        .ok_or(crate::BackendError::InvalidArgument {
                            operation: "device_block_operator_apply_colmajor",
                            reason: "block input view offset overflow".into(),
                        })?;
                    let destination_start = col.checked_mul(block_cols).ok_or(
                        crate::BackendError::InvalidArgument {
                            operation: "device_block_operator_apply_colmajor",
                            reason: "block input destination offset overflow".into(),
                        },
                    )?;
                    backend.try_copy_views_d2d(
                        x,
                        BufferView::new(x.len(), source_start, block_cols, 1)?,
                        x_dev,
                        BufferView::new(x_dev.len(), destination_start, block_cols, 1)?,
                    )?;
                }
                backend.try_copy_h2d(partial, y_dev)?;
                block.try_apply_colmajor(backend, rhs, x_dev, y_dev)?;
                backend.try_copy_d2h(y_dev, partial)?;

                for col in 0..rhs {
                    let dst_start = col * rows + row_range.start;
                    let src_start = col * block_rows;
                    for offset in 0..block_rows {
                        workspace.y_host[dst_start + offset] =
                            workspace.y_host[dst_start + offset] + partial[src_start + offset];
                    }
                }
            }
        }
        backend.try_copy_h2d(&workspace.y_host, y)
    }
}

impl<B: Backend, T: DeviceScalar> DeviceOperator<B, T> for DeviceBlockOperator<B, T> {
    fn n(&self) -> usize {
        self.layout.num_cols()
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, x, y) {
            log::error!(
                target: "sonium_compute::block_operator",
                "device block operator application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let mut workspace = self.new_workspace();
        self.try_apply_with_workspace(backend, x, y, &mut workspace)
    }

    fn try_apply_legacy(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let cols = self.layout.num_cols();
        let rows = self.layout.num_rows();
        if x.len() != cols || y.len() != rows {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_apply",
                reason: format!(
                    "expected input/output length {cols}/{rows}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }

        let mut x_host = zeroed_block_vec::<T>(x.len(), "device_block_operator input")?;
        let mut y_host = zeroed_block_vec::<T>(y.len(), "device_block_operator output")?;
        backend.try_copy_d2h(x, &mut x_host)?;

        let mut x_buffers: Vec<Option<B::Buffer<T>>> = Vec::new();
        reserve_block_vec(
            &mut x_buffers,
            self.layout.num_block_cols(),
            "device_block_operator input buffers",
        )?;
        x_buffers.resize_with(self.layout.num_block_cols(), || None);
        let mut y_buffers: Vec<Option<B::Buffer<T>>> = Vec::new();
        reserve_block_vec(
            &mut y_buffers,
            self.layout.num_block_rows(),
            "device_block_operator output buffers",
        )?;
        y_buffers.resize_with(self.layout.num_block_rows(), || None);
        let mut y_partials: Vec<Vec<T>> = Vec::new();
        reserve_block_vec(
            &mut y_partials,
            self.layout.num_block_rows(),
            "device_block_operator output partials",
        )?;
        for &row_size in &self.layout.row_sizes {
            y_partials.push(zeroed_block_vec::<T>(
                row_size,
                "device_block_operator output partial",
            )?);
        }

        for (block_row, y_partial) in y_partials.iter_mut().enumerate() {
            let row_range = self.layout.row_range(block_row);
            for (block_col, x_dev_slot) in x_buffers.iter_mut().enumerate() {
                let Some(block) = self.block(block_row, block_col) else {
                    continue;
                };
                let col_range = self.layout.col_range(block_col);
                if x_dev_slot.is_none() {
                    let x_dev = backend
                        .raw_alloc::<T>(col_range.len())
                        .map_err(crate::BackendError::from)?;
                    *x_dev_slot = Some(x_dev);
                }
                if y_buffers[block_row].is_none() {
                    let y_dev = backend
                        .raw_alloc::<T>(row_range.len())
                        .map_err(crate::BackendError::from)?;
                    y_buffers[block_row] = Some(y_dev);
                }

                let x_dev = x_dev_slot
                    .as_mut()
                    .ok_or(crate::BackendError::OperationFailed {
                        operation: "device_block_operator_apply",
                        reason: "input workspace allocation disappeared".to_string(),
                    })?;
                let y_dev =
                    y_buffers[block_row]
                        .as_mut()
                        .ok_or(crate::BackendError::OperationFailed {
                            operation: "device_block_operator_apply",
                            reason: "output workspace allocation disappeared".to_string(),
                        })?;

                backend.try_copy_h2d(&x_host[col_range.clone()], x_dev)?;
                for value in y_partial.iter_mut() {
                    *value = T::zero();
                }
                backend.try_copy_h2d(y_partial, y_dev)?;
                block.try_apply(backend, x_dev, y_dev)?;

                backend.try_copy_d2h(y_dev, y_partial)?;
                for (offset, value) in y_partial.iter().enumerate() {
                    y_host[row_range.start + offset] = y_host[row_range.start + offset] + *value;
                }
            }
        }

        backend.try_copy_h2d(&y_host, y)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, x, y) {
            log::error!(
                target: "sonium_compute::block_operator",
                "device block operator multi-RHS application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let mut workspace = self.new_workspace();
        self.try_apply_colmajor_with_workspace(backend, rhs, x, y, &mut workspace)
    }
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (*a - *e).abs() < 1.0e-12,
                "actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn block_layout_computes_offsets() {
        let layout = BlockLayout::new(vec![2, 3], vec![4, 1, 2]);
        assert_eq!(layout.num_block_rows(), 2);
        assert_eq!(layout.num_block_cols(), 3);
        assert_eq!(layout.num_rows(), 5);
        assert_eq!(layout.num_cols(), 7);
        assert_eq!(layout.row_range(1), 2..5);
        assert_eq!(layout.col_range(2), 5..7);
    }

    #[test]
    fn checked_block_layout_rejects_invalid_sizes_and_overflow() {
        assert!(matches!(
            BlockLayout::try_new(Vec::new(), vec![1]),
            Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                ..
            })
        ));
        assert!(matches!(
            BlockLayout::try_new(vec![0], vec![1]),
            Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                ..
            })
        ));
        assert!(matches!(
            BlockLayout::try_new(vec![usize::MAX, 1], vec![1]),
            Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                ..
            })
        ));
        assert!(matches!(
            BlockLayout::try_new(vec![1], Vec::new()),
            Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_new",
                ..
            })
        ));
    }

    #[test]
    fn checked_device_block_operator_rejects_invalid_global_and_local_shapes() {
        let rectangular = BlockLayout::try_new(vec![2], vec![3]).unwrap();
        assert!(matches!(
            DeviceBlockOperator::<CpuBackend, f64>::try_new(rectangular),
            Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_new",
                ..
            })
        ));

        let backend = CpuBackend::new();
        let layout = BlockLayout::try_new(vec![2], vec![2]).unwrap();
        assert!(matches!(
            layout.try_row_range(usize::MAX),
            Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_row",
                ..
            })
        ));
        assert!(matches!(
            layout.try_col_size(1),
            Err(crate::BackendError::InvalidArgument {
                operation: "block_layout_column",
                ..
            })
        ));
        let mut operator =
            DeviceBlockOperator::<CpuBackend, f64>::try_new(layout).expect("square layout");
        let local = DeviceCsr::from_host(&backend, 1, 1, &[0, 1], &[0], &[1.0]);
        assert!(matches!(
            operator.try_set_block(0, 0, Box::new(local)),
            Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_set_block",
                ..
            })
        ));
        assert!(matches!(
            operator.try_block(1, 0),
            Err(crate::BackendError::InvalidArgument {
                operation: "device_block_operator_block",
                ..
            })
        ));
    }

    #[test]
    fn device_block_operator_cpu_matches_host_block_result() {
        let backend = CpuBackend::new();
        let layout = BlockLayout::new(vec![2, 1], vec![2, 1]);

        let a00 = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 4.0],
        );
        let a01 = DeviceCsr::from_host(&backend, 2, 1, &[0, 1, 2], &[0, 0], &[5.0, 6.0]);
        let a10 = DeviceCsr::from_host(&backend, 1, 2, &[0, 2], &[0, 1], &[7.0, 8.0]);

        let op = DeviceBlockOperator::new(layout)
            .with_block(0, 0, Box::new(a00))
            .with_block(0, 1, Box::new(a01))
            .with_block(1, 0, Box::new(a10));

        let mut x = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[2.0, 3.0, 4.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(3).unwrap();

        op.apply(&backend, &x, &mut y);

        let mut actual = [0.0; 3];
        backend.copy_d2h(&y, &mut actual);
        assert_close(&actual, &[28.0, 42.0, 38.0]);
    }

    #[test]
    fn device_block_operator_apply_colmajor_matches_host_block_result() {
        let backend = CpuBackend::new();
        let layout = BlockLayout::new(vec![2, 1], vec![2, 1]);

        let a00 = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 4.0],
        );
        let a01 = DeviceCsr::from_host(&backend, 2, 1, &[0, 1, 2], &[0, 0], &[5.0, 6.0]);
        let a10 = DeviceCsr::from_host(&backend, 1, 2, &[0, 2], &[0, 1], &[7.0, 8.0]);

        let op = DeviceBlockOperator::new(layout)
            .with_block(0, 0, Box::new(a00))
            .with_block(0, 1, Box::new(a01))
            .with_block(1, 0, Box::new(a10));

        let mut x = backend.raw_alloc::<f64>(6).unwrap();
        backend.copy_h2d(&[2.0, 3.0, 4.0, 1.0, -1.0, 2.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(6).unwrap();

        op.try_apply_colmajor(&backend, 2, &x, &mut y)
            .expect("checked block apply must succeed on CPU");

        let mut actual = [0.0; 6];
        backend.copy_d2h(&y, &mut actual);
        assert_close(&actual, &[28.0, 42.0, 38.0, 9.0, 11.0, -1.0]);

        let input = [2.0, 3.0, 4.0, 1.0, -1.0, 2.0];
        let mut columnwise = [0.0; 6];
        for column in 0..2 {
            let mut x_column = backend.raw_alloc::<f64>(3).unwrap();
            let mut y_column = backend.raw_alloc::<f64>(3).unwrap();
            backend.copy_h2d(&input[column * 3..(column + 1) * 3], &mut x_column);
            op.try_apply(&backend, &x_column, &mut y_column)
                .expect("checked single-RHS block apply must succeed");
            backend.copy_d2h(&y_column, &mut columnwise[column * 3..(column + 1) * 3]);
        }
        assert_eq!(actual, columnwise);
    }

    struct PanicBlock;

    impl DeviceBlock<CpuBackend, f64> for PanicBlock {
        fn num_rows(&self) -> usize {
            1
        }

        fn num_cols(&self) -> usize {
            1
        }

        fn apply(
            &self,
            _backend: &CpuBackend,
            _x: &crate::cpu::HostBuffer<f64>,
            _y: &mut crate::cpu::HostBuffer<f64>,
        ) {
            panic!("simulated block backend failure");
        }
    }

    #[test]
    fn reusable_block_workspace_preserves_results_and_capacity() {
        let backend = CpuBackend::new();
        let layout = BlockLayout::new(vec![2, 1], vec![2, 1]);
        let a00 = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 4.0],
        );
        let a01 = DeviceCsr::from_host(&backend, 2, 1, &[0, 1, 2], &[0, 0], &[5.0, 6.0]);
        let a10 = DeviceCsr::from_host(&backend, 1, 2, &[0, 2], &[0, 1], &[7.0, 8.0]);
        let op = DeviceBlockOperator::new(layout.clone())
            .with_block(0, 0, Box::new(a00))
            .with_block(0, 1, Box::new(a01))
            .with_block(1, 0, Box::new(a10));

        let mut x = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[2.0, 3.0, 4.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(3).unwrap();
        let mut workspace = op.new_workspace();
        op.try_apply_with_workspace(&backend, &x, &mut y, &mut workspace)
            .expect("first reusable block application");
        let first_capacity = workspace.host_capacity();
        let first_buffers = workspace.device_buffer_count();
        assert!(first_capacity > 0);
        assert_eq!(first_buffers, 4);

        let mut actual = [0.0; 3];
        backend.copy_d2h(&y, &mut actual);
        assert_close(&actual, &[28.0, 42.0, 38.0]);

        backend.copy_h2d(&[1.0, -1.0, 2.0], &mut x);
        op.try_apply_with_workspace(&backend, &x, &mut y, &mut workspace)
            .expect("warm reusable block application");
        assert_eq!(workspace.host_capacity(), first_capacity);
        assert_eq!(workspace.device_buffer_count(), first_buffers);
        backend.copy_d2h(&y, &mut actual);
        assert_close(&actual, &[9.0, 11.0, -1.0]);

        workspace.clear();
        assert_eq!(workspace.device_buffer_count(), 0);
        op.try_apply_with_workspace(&backend, &x, &mut y, &mut workspace)
            .expect("reusable block application after clear");
        assert_eq!(workspace.device_buffer_count(), first_buffers);
    }

    #[test]
    fn reusable_block_workspace_rejects_a_different_layout() {
        let backend = CpuBackend::new();
        let op = DeviceBlockOperator::<CpuBackend, f64>::new(BlockLayout::new(vec![2], vec![2]));
        let other = BlockLayout::new(vec![1], vec![1]);
        let mut workspace = DeviceBlockWorkspace::<CpuBackend, f64>::new(&other);
        let x = backend.raw_alloc::<f64>(2).unwrap();
        let mut y = backend.raw_alloc::<f64>(2).unwrap();
        assert!(matches!(
            op.try_apply_with_workspace(&backend, &x, &mut y, &mut workspace),
            Err(crate::BackendError::InvalidArgument {
                operation: "device_block_workspace_layout",
                ..
            })
        ));
    }

    #[test]
    fn checked_generic_block_apply_contains_legacy_panics() {
        let backend = CpuBackend::new();
        let x = backend.raw_alloc::<f64>(1).unwrap();
        let mut y = backend.raw_alloc::<f64>(1).unwrap();
        let error = PanicBlock
            .try_apply(&backend, &x, &mut y)
            .expect_err("checked block apply must contain legacy panics");
        assert!(matches!(
            error,
            crate::BackendError::OperationFailed {
                operation: "device_block_apply",
                ..
            }
        ));
    }

    #[test]
    fn checked_generic_block_apply_propagates_pending_backend_errors() {
        struct CopyBlock;

        impl DeviceBlock<CpuBackend, f64> for CopyBlock {
            fn num_rows(&self) -> usize {
                1
            }

            fn num_cols(&self) -> usize {
                1
            }

            fn apply(
                &self,
                backend: &CpuBackend,
                x: &crate::cpu::HostBuffer<f64>,
                y: &mut crate::cpu::HostBuffer<f64>,
            ) {
                backend.copy_d2d(x, y);
            }
        }

        let backend = CpuBackend::new();
        let x = backend.raw_alloc::<f64>(1).unwrap();
        let mut y = backend.raw_alloc::<f64>(1).unwrap();
        CpuBackend::inject_pending_error();

        let error = CopyBlock
            .try_apply(&backend, &x, &mut y)
            .expect_err("checked block apply must surface pending backend errors");
        assert!(matches!(
            error,
            crate::BackendError::OperationFailed {
                operation: "cpu_pending_error",
                ..
            }
        ));
    }

    #[test]
    fn checked_device_csr_block_colmajor_contains_malformed_prebake() {
        let backend = CpuBackend::new();
        let mut row_ptr = backend.raw_alloc::<i32>(2).unwrap();
        let mut col_idx = backend.raw_alloc::<i32>(2).unwrap();
        let mut values = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[0, 2], &mut row_ptr);
        backend.copy_h2d(&[0, 1], &mut col_idx);
        backend.copy_h2d(&[1.0, 2.0], &mut values);
        let block = DeviceCsr {
            m: 2,
            n: 2,
            nnz: 2,
            row_ptr,
            col_idx,
            values,
        };
        let x = backend.raw_alloc::<f64>(4).unwrap();
        let mut y = backend.raw_alloc::<f64>(4).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <DeviceCsr<CpuBackend, f64> as DeviceBlock<CpuBackend, f64>>::try_apply_colmajor(
                &block, &backend, 2, &x, &mut y,
            )
        }));
        assert!(result.is_ok(), "checked block CSR path panicked");
        assert!(matches!(
            result.unwrap(),
            Err(crate::BackendError::InvalidArgument { .. })
                | Err(crate::BackendError::OperationFailed { .. })
        ));
    }

    #[test]
    fn square_device_block_wraps_arbitrary_device_operator() {
        let backend = CpuBackend::new();
        let csr = DeviceCsr::from_host(&backend, 2, 2, &[0, 1, 2], &[0, 1], &[2.0_f64, 3.0]);
        let block = SquareDeviceBlock::new(csr);

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[4.0, 5.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(2).unwrap();

        block.apply(&backend, &x, &mut y);

        let mut actual = [0.0; 2];
        backend.copy_d2h(&y, &mut actual);
        assert_close(&actual, &[8.0, 15.0]);
    }

    #[test]
    fn checked_device_block_operator_rejects_single_and_multi_rhs_shapes() {
        let backend = CpuBackend::new();
        let op = DeviceBlockOperator::<CpuBackend, f64>::new(BlockLayout::new(vec![2], vec![2]));

        let x = backend.raw_alloc::<f64>(1).unwrap();
        let mut y = backend.raw_alloc::<f64>(2).unwrap();
        assert!(matches!(
            op.try_apply(&backend, &x, &mut y),
            Err(crate::BackendError::InvalidArgument { .. })
        ));

        let x = backend.raw_alloc::<f64>(3).unwrap();
        let mut y = backend.raw_alloc::<f64>(4).unwrap();
        assert!(matches!(
            op.try_apply_colmajor(&backend, 2, &x, &mut y),
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "must be square")]
    fn device_block_operator_rejects_rectangular_global_system() {
        let _ = DeviceBlockOperator::<CpuBackend, f64>::new(BlockLayout::new(vec![2], vec![3]));
    }
}
