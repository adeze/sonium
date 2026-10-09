use num_complex::{Complex32, Complex64};
use num_traits::{Float, Zero};
use sonium_solvers::FgmresConfig;
use sonium_solvers::iterative::{FgmresSolution, KrylovOpCounts};
use sonium_solvers::mixed_precision::{
    MixedPrecisionCheckpoint, try_mixed_precision_solve, try_mixed_precision_solve_with_guess,
};
use sonium_solvers::preconditioners::DiagonalPreconditioner;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{AccuracyStatus, Preconditioner, SolverResultContract, SolverStatus};

fn real_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
    )
}

fn real_preconditioner(matrix: &CsrMatrix<f64>) -> Box<dyn Preconditioner<f32> + Send + Sync> {
    Box::new(DiagonalPreconditioner::from_csr(&matrix.cast::<f32>()))
}

fn relative_residual<T: sonium_solvers::traits::ComplexField>(
    matrix: &CsrMatrix<T>,
    x: &[T],
    rhs: &[T],
) -> T::Real {
    let action = matrix.matvec(x);
    action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(T::Real::zero(), |lhs, rhs| lhs.max(rhs))
}

#[test]
fn checkpoint_resume_preserves_metadata_and_cumulative_counts() {
    let matrix = real_matrix();
    let first_solution = [1.0, -2.0, 0.5];
    let second_solution = [-0.5, 1.25, 2.0];
    let first_rhs = matrix.matvec(&first_solution);
    let second_rhs = matrix.matvec(&second_solution);
    let config = FgmresConfig {
        max_iterations: 40,
        restart: 3,
        tolerance: 1.0e-12,
        exact_residual_check_interval: 1,
        ..FgmresConfig::default()
    };

    let first =
        try_mixed_precision_solve(&matrix, &first_rhs, real_preconditioner(&matrix), &config)
            .expect("first mixed-precision solve");
    first.validate_contract().expect("first result contract");
    assert!(first.converged);

    let checkpoint = MixedPrecisionCheckpoint::from_solution(&first)
        .expect("capture mixed-precision checkpoint");
    assert_eq!(checkpoint.dimension(), 3);
    assert_eq!(checkpoint.source_iterations(), Some(first.iterations));
    assert_eq!(checkpoint.source_restarts(), Some(first.restarts));
    assert_eq!(checkpoint.source_status(), Some(first.status));
    assert_eq!(
        checkpoint.source_accuracy_status(),
        Some(first.attainable_accuracy)
    );
    assert_eq!(checkpoint.source_op_counts(), Some(first.op_counts));
    assert!(checkpoint.storage_bytes() >= 3 * std::mem::size_of::<f64>());

    let resumed = checkpoint
        .resume(&matrix, &second_rhs, real_preconditioner(&matrix), &config)
        .expect("resume mixed-precision solve");
    resumed
        .validate_contract()
        .expect("resumed result contract");
    assert!(resumed.converged);
    assert!(resumed.iterations >= first.iterations);
    assert!(resumed.op_counts.spmv_count >= first.op_counts.spmv_count);
    assert!(relative_residual(&matrix, &resumed.x, &second_rhs) < 1.0e-10);
    for (actual, expected) in resumed.x.iter().zip(second_solution) {
        assert!((*actual - expected).abs() < 1.0e-9);
    }
}

#[test]
fn iterate_checkpoint_and_explicit_initial_guess_are_checked() {
    let matrix = real_matrix();
    let expected = [0.75, -1.5, 1.25];
    let rhs = matrix.matvec(&expected);
    let config = FgmresConfig {
        max_iterations: 40,
        restart: 3,
        tolerance: 1.0e-12,
        ..FgmresConfig::default()
    };
    let checkpoint =
        MixedPrecisionCheckpoint::from_iterate(&[0.0, 0.0, 0.0]).expect("capture initial iterate");
    assert_eq!(checkpoint.source_iterations(), None);
    assert_eq!(checkpoint.source_residual(), None);
    let resumed = checkpoint
        .resume(&matrix, &rhs, real_preconditioner(&matrix), &config)
        .expect("resume from initial iterate");
    assert!(resumed.converged);
    assert!(relative_residual(&matrix, &resumed.x, &rhs) < 1.0e-10);

    let explicit = try_mixed_precision_solve_with_guess(
        &matrix,
        &rhs,
        Some(&[0.0, 0.0, 0.0]),
        real_preconditioner(&matrix),
        &config,
    )
    .expect("checked explicit initial guess");
    assert!(explicit.converged);
    assert!(relative_residual(&matrix, &explicit.x, &rhs) < 1.0e-10);
}

#[test]
fn checkpoint_rejects_nonfinite_iterates_and_invalid_result_metadata() {
    assert!(MixedPrecisionCheckpoint::<f64>::from_iterate(&[0.0, f64::NAN,]).is_err());
    let invalid = FgmresSolution {
        x: vec![0.0_f64; 2],
        iterations: 1,
        restarts: 0,
        residual: f64::NAN,
        estimated_residual: 1.0,
        backward_error: 1.0,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts: KrylovOpCounts::default(),
    };
    assert!(MixedPrecisionCheckpoint::from_solution(&invalid).is_err());
}

#[test]
fn complex_mixed_precision_checkpoint_round_trip() {
    let matrix = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(3.0, 0.25),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(2.5, -0.2),
        ],
    );
    let expected = [Complex64::new(1.0, 0.5), Complex64::new(-0.75, 0.25)];
    let rhs = matrix.matvec(&expected);
    let config = FgmresConfig {
        max_iterations: 30,
        restart: 2,
        tolerance: 1.0e-11,
        ..FgmresConfig::default()
    };
    let low_matrix = matrix.cast::<Complex32>();
    let first = try_mixed_precision_solve(
        &matrix,
        &rhs,
        Box::new(DiagonalPreconditioner::from_csr(&low_matrix)),
        &config,
    )
    .expect("complex mixed-precision solve");
    let checkpoint =
        MixedPrecisionCheckpoint::from_solution(&first).expect("capture complex checkpoint");
    let resumed = checkpoint
        .resume(
            &matrix,
            &rhs,
            Box::new(DiagonalPreconditioner::from_csr(&low_matrix)),
            &config,
        )
        .expect("resume complex checkpoint");
    assert!(resumed.converged);
    assert!(relative_residual(&matrix, &resumed.x, &rhs) < 1.0e-9);
    for (actual, expected) in resumed.x.iter().zip(expected) {
        assert!((*actual - expected).norm() < 1.0e-8);
    }
}

fn metadata_checkpoint() -> MixedPrecisionCheckpoint<f64> {
    let solution = FgmresSolution {
        x: vec![1.25, -2.5],
        iterations: 7,
        restarts: 2,
        residual: 1.0e-8,
        estimated_residual: 2.0e-8,
        backward_error: 3.0e-9,
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::Converged,
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
    MixedPrecisionCheckpoint::from_solution(&solution).unwrap()
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn portable_checkpoint_round_trip_preserves_real_metadata() {
    let checkpoint = metadata_checkpoint();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    assert_eq!(
        &bytes[..sonium_solvers::MIXED_PRECISION_CHECKPOINT_MAGIC.len()],
        sonium_solvers::MIXED_PRECISION_CHECKPOINT_MAGIC
    );
    let restored = MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.solution(), checkpoint.solution());
    assert_eq!(restored.source_iterations(), Some(7));
    assert_eq!(restored.source_restarts(), Some(2));
    assert_eq!(restored.source_residual(), Some(1.0e-8));
    assert_eq!(restored.source_estimated_residual(), Some(2.0e-8));
    assert_eq!(restored.source_backward_error(), Some(3.0e-9));
    assert_eq!(
        restored.source_accuracy_status(),
        Some(AccuracyStatus::Attained)
    );
    assert_eq!(restored.source_converged(), Some(true));
    assert_eq!(restored.source_status(), Some(SolverStatus::Converged));
    assert_eq!(restored.source_op_counts(), checkpoint.source_op_counts());
}

#[test]
fn portable_checkpoint_round_trip_preserves_complex_metadata() {
    let solution = FgmresSolution {
        x: vec![Complex64::new(1.25, -0.5), Complex64::new(-2.5, 0.75)],
        iterations: 3,
        restarts: 1,
        residual: 4.0e-7,
        estimated_residual: 5.0e-7,
        backward_error: 6.0e-8,
        attainable_accuracy: AccuracyStatus::NotAttained,
        converged: false,
        status: SolverStatus::MaxIterationsReached,
        op_counts: KrylovOpCounts::default(),
    };
    let checkpoint = MixedPrecisionCheckpoint::from_solution(&solution).unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored = MixedPrecisionCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.solution(), checkpoint.solution());
    assert_eq!(restored.source_iterations(), Some(3));
    assert_eq!(
        restored.source_status(),
        Some(SolverStatus::MaxIterationsReached)
    );
    assert_eq!(restored.source_converged(), Some(false));
    assert_eq!(
        restored.source_accuracy_status(),
        Some(AccuracyStatus::NotAttained)
    );
}

#[test]
fn portable_iterate_only_checkpoint_round_trip_has_no_metadata() {
    let checkpoint = MixedPrecisionCheckpoint::<f64>::from_iterate(&[0.25, -1.5]).unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored = MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.solution(), &[0.25, -1.5]);
    assert_eq!(restored.dimension(), 2);
    assert_eq!(restored.storage_bytes(), checkpoint.storage_bytes());
    assert_eq!(restored.source_iterations(), None);
    assert_eq!(restored.source_op_counts(), None);
}

#[test]
fn portable_checkpoint_rejects_header_size_and_structure_corruption() {
    let bytes = metadata_checkpoint().to_checkpoint_bytes().unwrap();

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_magic).is_err());

    let mut bad_version = bytes.clone();
    bad_version[8..10].copy_from_slice(
        &(sonium_solvers::MIXED_PRECISION_CHECKPOINT_FORMAT_VERSION + 1).to_le_bytes(),
    );
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_version).is_err());

    let mut bad_scalar_family = bytes.clone();
    assert!(MixedPrecisionCheckpoint::<f32>::from_checkpoint_bytes(&bad_scalar_family).is_err());

    bad_scalar_family[13] = 2;
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_scalar_family).is_err());

    let mut bad_storage_size = bytes.clone();
    put_u64(&mut bad_storage_size, 22, 0);
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_storage_size).is_err());

    let mut bad_dimension = bytes.clone();
    put_u64(&mut bad_dimension, 14, u64::MAX);
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_dimension).is_err());

    assert!(
        MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err()
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&trailing).is_err());
}

#[test]
fn portable_checkpoint_rejects_invalid_scalar_and_diagnostic_tags() {
    let bytes = metadata_checkpoint().to_checkpoint_bytes().unwrap();

    let mut bad_scalar = bytes.clone();
    bad_scalar[30..38].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_scalar).is_err());

    let mut bad_real_imaginary = bytes.clone();
    bad_real_imaginary[38..46].copy_from_slice(&1.0f64.to_le_bytes());
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_real_imaginary).is_err());

    // Header is 30 bytes and each scalar occupies a canonical 16-byte pair.
    let metrics = 30 + 2 * 16;
    let mut bad_diagnostic = bytes.clone();
    bad_diagnostic[metrics + 16..metrics + 24].copy_from_slice(&(-1.0f64).to_le_bytes());
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_diagnostic).is_err());

    let mut bad_accuracy = bytes.clone();
    bad_accuracy[metrics + 40] = 99;
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_accuracy).is_err());

    let mut bad_convergence = bytes.clone();
    bad_convergence[metrics + 41] = 0;
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_convergence).is_err());

    let mut bad_status = bytes.clone();
    bad_status[metrics + 42] = 99;
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_status).is_err());

    let mut bad_reserved = bytes;
    bad_reserved[metrics + 43] = 1;
    assert!(MixedPrecisionCheckpoint::<f64>::from_checkpoint_bytes(&bad_reserved).is_err());
}
