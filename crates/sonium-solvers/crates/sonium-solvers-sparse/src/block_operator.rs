//! Backend-neutral block linear operators.

pub use sonium_compute::BlockLayout;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, LinearOperator};
use std::marker::PhantomData;

/// CSR blocks already satisfy [`LinearOperator`]; this alias names that role
/// in coupled block systems.
pub type BlockCsr<T> = crate::CsrMatrix<T>;

fn combine_adjoint_capabilities(
    left: AdjointCapability,
    right: AdjointCapability,
) -> AdjointCapability {
    if left == AdjointCapability::Unsupported || right == AdjointCapability::Unsupported {
        AdjointCapability::Unsupported
    } else if left == AdjointCapability::Unknown || right == AdjointCapability::Unknown {
        AdjointCapability::Unknown
    } else {
        AdjointCapability::Exact
    }
}

pub struct BlockOperator<T: ComplexField> {
    layout: BlockLayout,
    blocks: Vec<Option<Box<dyn LinearOperator<T>>>>,
}

impl<T: ComplexField> BlockOperator<T> {
    pub fn new(layout: BlockLayout) -> Self {
        let block_count = layout.num_block_rows() * layout.num_block_cols();
        Self {
            layout,
            blocks: (0..block_count).map(|_| None).collect(),
        }
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn set_block(
        &mut self,
        block_row: usize,
        block_col: usize,
        block: Box<dyn LinearOperator<T>>,
    ) {
        self.check_block_shape(block_row, block_col, block.as_ref());
        let index = self.block_index(block_row, block_col);
        self.blocks[index] = Some(block);
    }

    pub fn with_block(
        mut self,
        block_row: usize,
        block_col: usize,
        block: Box<dyn LinearOperator<T>>,
    ) -> Self {
        self.set_block(block_row, block_col, block);
        self
    }

    pub fn block(&self, block_row: usize, block_col: usize) -> Option<&dyn LinearOperator<T>> {
        self.blocks[self.block_index(block_row, block_col)].as_deref()
    }

    /// Maximum scratch lengths needed by [`Self::apply_into_with_scratch`]:
    /// `(max_block_cols, max_block_rows)` over all block rows/cols.
    pub fn scratch_lens(&self) -> (usize, usize) {
        let max_cols = (0..self.layout.num_block_cols())
            .map(|c| self.layout.col_size(c))
            .max()
            .unwrap_or(0);
        let max_rows = (0..self.layout.num_block_rows())
            .map(|r| self.layout.row_size(r))
            .max()
            .unwrap_or(0);
        (max_cols, max_rows)
    }

    /// [`LinearOperator::apply_into`] without per-block allocation: `x_scratch`
    /// must hold the widest block column (`scratch_lens().0`) and `p_scratch`
    /// the tallest block row (`scratch_lens().1`). Both are overwritten.
    pub fn apply_into_with_scratch(
        &self,
        x: &[T],
        y: &mut [T],
        x_scratch: &mut [T],
        p_scratch: &mut [T],
    ) {
        assert_eq!(
            x.len(),
            self.num_cols(),
            "block operator input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.num_rows(),
            "block operator output length mismatch"
        );
        let (need_x, need_p) = self.scratch_lens();
        assert!(x_scratch.len() >= need_x, "x scratch too small");
        assert!(p_scratch.len() >= need_p, "partial scratch too small");
        for v in y.iter_mut() {
            *v = T::zero();
        }

        for block_row in 0..self.layout.num_block_rows() {
            let row_range = self.layout.row_range(block_row);
            for block_col in 0..self.layout.num_block_cols() {
                let Some(block) = self.block(block_row, block_col) else {
                    continue;
                };
                let col_range = self.layout.col_range(block_col);
                let x_block = &mut x_scratch[..col_range.len()];
                x_block.copy_from_slice(&x[col_range.clone()]);
                let partial = &mut p_scratch[..row_range.len()];
                block.apply_into(x_block, partial);
                for (offset, value) in partial.iter().enumerate() {
                    y[row_range.start + offset] += *value;
                }
            }
        }
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

    fn check_block_shape(&self, block_row: usize, block_col: usize, block: &dyn LinearOperator<T>) {
        let expected_rows = self.layout.row_size(block_row);
        let expected_cols = self.layout.col_size(block_col);
        assert_eq!(
            block.num_rows(),
            expected_rows,
            "block ({block_row}, {block_col}) row count mismatch"
        );
        assert_eq!(
            block.num_cols(),
            expected_cols,
            "block ({block_row}, {block_col}) column count mismatch"
        );
    }
}

impl<T: ComplexField> LinearOperator<T> for BlockOperator<T> {
    fn num_rows(&self) -> usize {
        self.layout.num_rows()
    }
    fn num_cols(&self) -> usize {
        self.layout.num_cols()
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.num_cols(),
            "block operator input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.num_rows(),
            "block operator output length mismatch"
        );
        for v in y.iter_mut() {
            *v = T::zero();
        }

        for block_row in 0..self.layout.num_block_rows() {
            let row_range = self.layout.row_range(block_row);
            for block_col in 0..self.layout.num_block_cols() {
                let Some(block) = self.block(block_row, block_col) else {
                    continue;
                };
                let col_range = self.layout.col_range(block_col);
                let x_block: Vec<T> = col_range.clone().map(|index| x[index]).collect();
                let partial = block.apply(&x_block);
                assert_eq!(partial.len(), row_range.len());
                for (offset, value) in partial.iter().enumerate() {
                    y[row_range.start + offset] += *value;
                }
            }
        }
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.num_rows(),
            "block transpose input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.num_cols(),
            "block transpose output length mismatch"
        );
        for v in y.iter_mut() {
            *v = T::zero();
        }

        for block_row in 0..self.layout.num_block_rows() {
            let row_range = self.layout.row_range(block_row);
            let x_block: Vec<T> = row_range.clone().map(|index| x[index]).collect();
            for block_col in 0..self.layout.num_block_cols() {
                let Some(block) = self.block(block_row, block_col) else {
                    continue;
                };
                let col_range = self.layout.col_range(block_col);
                let partial = block.apply_transpose(&x_block);
                assert_eq!(partial.len(), col_range.len());
                for (offset, value) in partial.iter().enumerate() {
                    y[col_range.start + offset] += *value;
                }
            }
        }
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.num_rows(),
            "block Hermitian input length mismatch"
        );
        let mut y = vec![T::zero(); self.num_cols()];

        for block_row in 0..self.layout.num_block_rows() {
            let row_range = self.layout.row_range(block_row);
            let x_block: Vec<T> = row_range.clone().map(|index| x[index]).collect();
            for block_col in 0..self.layout.num_block_cols() {
                let Some(block) = self.block(block_row, block_col) else {
                    continue;
                };
                let col_range = self.layout.col_range(block_col);
                let partial = block.apply_hermitian(&x_block);
                assert_eq!(partial.len(), col_range.len());
                for (offset, value) in partial.iter().enumerate() {
                    y[col_range.start + offset] += *value;
                }
            }
        }

        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.blocks
            .iter()
            .filter_map(|block| block.as_deref())
            .map(LinearOperator::transpose_capability)
            .fold(AdjointCapability::Exact, combine_adjoint_capabilities)
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.blocks
            .iter()
            .filter_map(|block| block.as_deref())
            .map(LinearOperator::hermitian_capability)
            .fold(AdjointCapability::Exact, combine_adjoint_capabilities)
    }
}

pub struct DenseBlockOperator<T: ComplexField> {
    matrix: DenseMatrix<T>,
}

impl<T: ComplexField> DenseBlockOperator<T> {
    pub fn new(matrix: DenseMatrix<T>) -> Self {
        Self { matrix }
    }
    pub fn matrix(&self) -> &DenseMatrix<T> {
        &self.matrix
    }
}

impl<T: ComplexField> LinearOperator<T> for DenseBlockOperator<T> {
    fn num_rows(&self) -> usize {
        self.matrix.nrows()
    }
    fn num_cols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.num_cols(),
            "dense block input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.num_rows(),
            "dense block output length mismatch"
        );
        self.matrix.matvec_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.num_rows(),
            "dense transpose input length mismatch"
        );
        assert_eq!(
            y.len(),
            self.num_cols(),
            "dense transpose output length mismatch"
        );
        self.matrix.matvec_transpose_into(x, y);
    }

    fn apply_hermitian(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.num_rows(),
            "dense Hermitian input length mismatch"
        );
        let mut y = vec![T::zero(); self.num_cols()];
        self.matrix.matvec_hermitian_into(x, &mut y);
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

pub struct ZeroBlock<T: ComplexField> {
    rows: usize,
    cols: usize,
    _scalar: PhantomData<T>,
}

impl<T: ComplexField> ZeroBlock<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            _scalar: PhantomData,
        }
    }
}

impl<T: ComplexField> LinearOperator<T> for ZeroBlock<T> {
    fn num_rows(&self) -> usize {
        self.rows
    }
    fn num_cols(&self) -> usize {
        self.cols
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.cols, "zero block input length mismatch");
        assert_eq!(y.len(), self.rows, "zero block output length mismatch");
        for v in y.iter_mut() {
            *v = T::zero();
        }
    }

    fn apply_transpose_into(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.rows, "zero transpose input length mismatch");
        assert_eq!(y.len(), self.cols, "zero transpose output length mismatch");
        for v in y.iter_mut() {
            *v = T::zero();
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn dm<T: ComplexField>(rows: usize, cols: usize, data: Vec<T>) -> DenseMatrix<T> {
        DenseMatrix::from_row_major_vec(rows, cols, data)
    }

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
        assert_eq!(layout.num_rows(), 5);
        assert_eq!(layout.num_cols(), 7);
        assert_eq!(layout.row_range(1), 2..5);
        assert_eq!(layout.col_range(2), 5..7);
    }

    #[test]
    fn dense_block_operator_rectangular_apply_and_transpose() {
        let block = DenseBlockOperator::new(dm(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        assert_close(&block.apply(&[7.0, 8.0, 9.0]), &[50.0, 122.0]);
        assert_close(&block.apply_transpose(&[2.0, 3.0]), &[14.0, 19.0, 24.0]);
    }

    #[test]
    fn block_operator_apply_matches_explicit_dense() {
        let layout = BlockLayout::new(vec![2, 1], vec![2, 1]);
        let op = BlockOperator::new(layout)
            .with_block(
                0,
                0,
                Box::new(DenseBlockOperator::new(dm(2, 2, vec![1.0, 2.0, 3.0, 4.0]))),
            )
            .with_block(
                0,
                1,
                Box::new(DenseBlockOperator::new(dm(2, 1, vec![5.0, 6.0]))),
            )
            .with_block(
                1,
                0,
                Box::new(DenseBlockOperator::new(dm(1, 2, vec![7.0, 8.0]))),
            );

        let y = op.apply(&[2.0, 3.0, 4.0]);
        assert_close(&y, &[28.0, 42.0, 38.0]);
    }

    #[test]
    fn block_operator_transpose_matches_explicit_dense() {
        let layout = BlockLayout::new(vec![2, 1], vec![2, 1]);
        let op = BlockOperator::new(layout)
            .with_block(
                0,
                0,
                Box::new(DenseBlockOperator::new(dm(2, 2, vec![1.0, 2.0, 3.0, 4.0]))),
            )
            .with_block(
                0,
                1,
                Box::new(DenseBlockOperator::new(dm(2, 1, vec![5.0, 6.0]))),
            )
            .with_block(
                1,
                0,
                Box::new(DenseBlockOperator::new(dm(1, 2, vec![7.0, 8.0]))),
            );

        let y = op.apply_transpose(&[2.0, 3.0, 4.0]);
        assert_close(&y, &[39.0, 48.0, 28.0]);
    }

    #[test]
    fn block_operator_hermitian_conjugates_blocks() {
        let layout = BlockLayout::new(vec![1], vec![2]);
        let op = BlockOperator::new(layout).with_block(
            0,
            0,
            Box::new(DenseBlockOperator::new(dm(
                1,
                2,
                vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
            ))),
        );

        let y = op.apply_hermitian(&[Complex64::new(2.0, 0.0)]);
        assert_eq!(y[0], Complex64::new(2.0, -4.0));
        assert_eq!(y[1], Complex64::new(6.0, 8.0));
        assert_eq!(op.hermitian_capability(), AdjointCapability::Exact);
        assert_eq!(
            op.try_apply_hermitian(&[Complex64::new(2.0, 0.0)])
                .expect("exact block Hermitian action should be checked-capable"),
            y
        );
    }

    #[test]
    #[should_panic(expected = "row count mismatch")]
    fn block_operator_rejects_shape_mismatch() {
        let layout = BlockLayout::new(vec![2], vec![2]);
        let _ = BlockOperator::new(layout).with_block(
            0,
            0,
            Box::new(DenseBlockOperator::new(dm(1, 2, vec![1.0, 2.0]))),
        );
    }

    #[test]
    fn zero_block_is_neutral() {
        let block = ZeroBlock::<f64>::new(2, 3);
        assert_close(&block.apply(&[1.0, 2.0, 3.0]), &[0.0, 0.0]);
        assert_close(&block.apply_transpose(&[1.0, 2.0]), &[0.0, 0.0, 0.0]);
    }
}
