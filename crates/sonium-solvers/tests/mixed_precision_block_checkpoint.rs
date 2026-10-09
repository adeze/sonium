use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, Zero};
use sonium_solvers::BlockFgmresConfig;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::KrylovOpCounts;
use sonium_solvers::mixed_precision::{
    MixedPrecisionBlockCheckpoint, try_mixed_precision_block_solve,
};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{ComplexField, Preconditioner, SolverError};

fn matrix<T: ComplexField>() -> CsrMatrix<T> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            T::from_real(T::Real::from_f64(4.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(4.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(4.0).unwrap()),
        ],
    )
}

fn block_from_columns<T: ComplexField>(columns: &[[T; 3]; 2]) -> DenseMatrix<T> {
    let mut block = DenseMatrix::from_elem(3, 2, T::zero());
    for column in 0..2 {
        for row in 0..3 {
            block[[row, column]] = columns[column][row];
        }
    }
    block
}

fn rhs<T: ComplexField>(matrix: &CsrMatrix<T>, columns: &[[T; 3]; 2]) -> DenseMatrix<T> {
    let mut block = DenseMatrix::from_elem(3, 2, T::zero());
    for column in 0..2 {
        let values = matrix.matvec(&columns[column]);
        for row in 0..3 {
            block[[row, column]] = values[row];
        }
    }
    block
}

fn real_preconditioner(matrix: &CsrMatrix<f64>) -> Box<dyn Preconditioner<f32> + Send + Sync> {
    Box::new(DiagonalPreconditioner::from_csr(&matrix.cast::<f32>()))
}

fn complex_preconditioner(
    matrix: &CsrMatrix<Complex64>,
) -> Box<dyn Preconditioner<Complex32> + Send + Sync> {
    Box::new(DiagonalPreconditioner::from_csr(
        &matrix.cast::<Complex32>(),
    ))
}

fn config() -> BlockFgmresConfig<f64> {
    BlockFgmresConfig {
        max_iterations: 20,
        restart: 3,
        tolerance: 1.0e-11,
        true_residual_check: true,
        exact_residual_check_interval: 1,
    }
}

fn max_error<T: ComplexField>(actual: &DenseMatrix<T>, expected: &DenseMatrix<T>) -> T::Real {
    actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(T::Real::zero(), |left, right| left.max(right))
}

#[test]
fn block_checkpoint_resumes_and_accumulates_metadata() {
    let operator = matrix::<f64>();
    let first_columns = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5]];
    let second_columns = [[2.0, 0.25, -1.0], [1.25, -1.5, 0.75]];
    let first_rhs = rhs(&operator, &first_columns);
    let second_rhs = rhs(&operator, &second_columns);
    let expected_second = block_from_columns(&second_columns);

    let first = try_mixed_precision_block_solve(
        &operator,
        &first_rhs,
        real_preconditioner(&operator),
        &config(),
    )
    .expect("first block solve");
    assert!(first.converged);
    let checkpoint = MixedPrecisionBlockCheckpoint::from_solution(&first)
        .expect("finite block result should checkpoint");
    assert_eq!(checkpoint.dimension(), 3);
    assert_eq!(checkpoint.num_rhs(), 2);
    assert_eq!(checkpoint.source_iterations(), Some(first.iterations));
    assert_eq!(checkpoint.source_converged(), Some(first.converged));
    assert_eq!(
        checkpoint.source_residuals(),
        Some(first.residuals.as_slice())
    );
    assert_eq!(
        checkpoint.source_true_residuals(),
        Some(first.true_residuals.as_slice())
    );
    assert_eq!(
        checkpoint.source_backward_errors(),
        Some(first.backward_errors.as_slice())
    );
    assert_eq!(
        checkpoint.source_attainable_accuracy(),
        Some(first.attainable_accuracy.as_slice())
    );
    assert!(checkpoint.storage_bytes() >= 6 * std::mem::size_of::<f64>());

    let resumed = checkpoint
        .resume(
            &operator,
            &second_rhs,
            real_preconditioner(&operator),
            &config(),
        )
        .expect("block checkpoint resume");
    let fresh = try_mixed_precision_block_solve(
        &operator,
        &second_rhs,
        real_preconditioner(&operator),
        &config(),
    )
    .expect("fresh block solve");
    assert!(resumed.converged);
    assert!(fresh.converged);
    assert!(max_error(&resumed.x, &expected_second) < 1.0e-9);
    assert!(max_error(&resumed.x, &fresh.x) < 1.0e-9);
    assert!(resumed.iterations >= first.iterations);
    assert!(resumed.op_counts.spmv_count >= first.op_counts.spmv_count);
}

#[test]
fn block_iterate_checkpoint_supports_explicit_initial_block() {
    let operator = matrix::<f64>();
    let columns = [[1.0, -1.0, 2.0], [-0.5, 2.0, 1.5]];
    let block = block_from_columns(&columns);
    let rhs = rhs(&operator, &columns);
    let checkpoint = MixedPrecisionBlockCheckpoint::from_iterate(&block).unwrap();
    assert_eq!(checkpoint.source_iterations(), None);
    assert_eq!(checkpoint.source_residuals(), None);
    assert_eq!(checkpoint.source_true_residuals(), None);
    assert_eq!(checkpoint.source_backward_errors(), None);
    assert_eq!(checkpoint.source_attainable_accuracy(), None);
    let result = checkpoint
        .resume(&operator, &rhs, real_preconditioner(&operator), &config())
        .unwrap();
    assert!(result.converged);
    assert!(max_error(&result.x, &block) < 1.0e-9);
}

#[test]
fn block_checkpoint_rejects_nonfinite_and_malformed_metadata() {
    let bad = DenseMatrix::from_elem(2, 2, f64::NAN);
    assert!(MixedPrecisionBlockCheckpoint::from_iterate(&bad).is_err());

    let malformed = sonium_solvers::BlockFgmresSolution {
        x: DenseMatrix::from_elem(2, 2, 0.0),
        iterations: 0,
        residuals: vec![0.0],
        true_residuals: vec![0.0, 0.0],
        backward_errors: vec![0.0, 0.0],
        attainable_accuracy: vec![],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    let error = MixedPrecisionBlockCheckpoint::from_solution(&malformed)
        .expect_err("per-RHS metadata mismatch must reject");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "mixed-precision block checkpoint",
            ..
        }
    ));
}

#[test]
fn complex_block_checkpoint_preserves_high_precision_solution() {
    let operator = CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.2),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.0, -0.2),
            Complex64::new(-1.0, 0.05),
            Complex64::new(-1.0, -0.05),
            Complex64::new(4.0, 0.1),
        ],
    );
    let columns = [
        [
            Complex64::new(1.0, 0.5),
            Complex64::new(-1.0, 0.25),
            Complex64::new(2.0, -0.5),
        ],
        [
            Complex64::new(-0.5, 0.1),
            Complex64::new(2.0, -0.2),
            Complex64::new(1.5, 0.3),
        ],
    ];
    let rhs = rhs(&operator, &columns);
    let result = try_mixed_precision_block_solve(
        &operator,
        &rhs,
        complex_preconditioner(&operator),
        &config(),
    )
    .expect("complex block solve");
    assert!(result.converged);
    let checkpoint = MixedPrecisionBlockCheckpoint::from_solution(&result).unwrap();
    assert_eq!(checkpoint.num_rhs(), 2);
    assert!(
        max_error(
            &checkpoint.solution().clone(),
            &block_from_columns(&columns)
        ) < 1.0e-5
    );
}

fn metadata_block_checkpoint() -> MixedPrecisionBlockCheckpoint<f64> {
    let solution = sonium_solvers::BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(2, 2, vec![1.25, -2.5, 0.75, 3.0]),
        iterations: 7,
        residuals: vec![1.0e-8, 2.0e-8],
        true_residuals: vec![3.0e-8, 4.0e-8],
        backward_errors: vec![5.0e-9, 6.0e-9],
        attainable_accuracy: vec![
            sonium_solvers::traits::AccuracyStatus::Attained,
            sonium_solvers::traits::AccuracyStatus::NotAttained,
        ],
        converged: true,
        op_counts: KrylovOpCounts {
            spmv_count: 11,
            preconditioner_count: 12,
            dot_reductions: 13,
            norm_count: 14,
            axpy_passes: 15,
            basis_bytes_touched: 16,
            synchronization_points: 17,
            exact_residual_checks: 18,
        },
    };
    MixedPrecisionBlockCheckpoint::from_solution(&solution).unwrap()
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn portable_block_checkpoint_round_trip_preserves_real_metadata() {
    let checkpoint = metadata_block_checkpoint();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored = MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(
        restored.solution().as_slice(),
        checkpoint.solution().as_slice()
    );
    assert_eq!(restored.dimension(), 2);
    assert_eq!(restored.num_rhs(), 2);
    assert_eq!(restored.source_iterations(), Some(7));
    assert_eq!(restored.source_residuals(), checkpoint.source_residuals());
    assert_eq!(
        restored.source_true_residuals(),
        checkpoint.source_true_residuals()
    );
    assert_eq!(
        restored.source_backward_errors(),
        checkpoint.source_backward_errors()
    );
    assert_eq!(
        restored.source_attainable_accuracy(),
        checkpoint.source_attainable_accuracy()
    );
    assert_eq!(restored.source_converged(), Some(true));
    assert_eq!(restored.source_op_counts(), checkpoint.source_op_counts());
}

#[test]
fn portable_block_checkpoint_round_trip_preserves_complex_iterate() {
    let checkpoint = MixedPrecisionBlockCheckpoint::from_iterate(&DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, -0.25),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.75, 0.125),
            Complex64::new(3.0, -0.75),
        ],
    ))
    .unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored =
        MixedPrecisionBlockCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(
        restored.solution().as_slice(),
        checkpoint.solution().as_slice()
    );
    assert_eq!(restored.source_iterations(), None);
    assert_eq!(restored.source_op_counts(), None);
}

#[test]
fn portable_block_checkpoint_rejects_header_shape_and_payload_corruption() {
    let bytes = metadata_block_checkpoint().to_checkpoint_bytes().unwrap();

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_magic).is_err());

    let mut bad_version = bytes.clone();
    bad_version[8..10].copy_from_slice(
        &(sonium_solvers::MIXED_PRECISION_BLOCK_CHECKPOINT_FORMAT_VERSION + 1).to_le_bytes(),
    );
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_version).is_err());

    assert!(MixedPrecisionBlockCheckpoint::<f32>::from_checkpoint_bytes(&bytes).is_err());

    let mut bad_flag = bytes.clone();
    bad_flag[13] = 2;
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_flag).is_err());

    let mut bad_rows = bytes.clone();
    put_u64(&mut bad_rows, 14, u64::MAX);
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_rows).is_err());

    let mut bad_storage = bytes.clone();
    put_u64(&mut bad_storage, 30, 0);
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_storage).is_err());

    assert!(
        MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1])
            .is_err()
    );
    let mut trailing = bytes;
    trailing.push(0);
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&trailing).is_err());
}

#[test]
fn portable_block_checkpoint_rejects_invalid_scalar_and_metadata_tags() {
    let bytes = metadata_block_checkpoint().to_checkpoint_bytes().unwrap();
    let metrics = 38 + 4 * 16;

    let mut bad_scalar = bytes.clone();
    bad_scalar[38..46].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_scalar).is_err());

    let mut bad_real_imag = bytes.clone();
    bad_real_imag[46..54].copy_from_slice(&1.0f64.to_le_bytes());
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_real_imag).is_err());

    let mut bad_diagnostic = bytes.clone();
    bad_diagnostic[metrics + 8..metrics + 16].copy_from_slice(&(-1.0f64).to_le_bytes());
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_diagnostic).is_err());

    let mut bad_accuracy = bytes.clone();
    bad_accuracy[metrics + 56] = 99;
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_accuracy).is_err());

    let mut bad_convergence = bytes.clone();
    bad_convergence[metrics + 58] = 2;
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_convergence).is_err());

    let mut bad_reserved = bytes;
    bad_reserved[metrics + 59] = 1;
    assert!(MixedPrecisionBlockCheckpoint::<f64>::from_checkpoint_bytes(&bad_reserved).is_err());
}
