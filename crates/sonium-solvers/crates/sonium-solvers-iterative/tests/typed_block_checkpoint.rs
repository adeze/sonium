use num_complex::{Complex32, Complex64};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::AccuracyStatus;
use sonium_solvers_iterative::{BlockFgmresSolution, BlockGmresSolution, KrylovOpCounts};

fn counts() -> KrylovOpCounts {
    KrylovOpCounts {
        spmv_count: 3,
        preconditioner_count: 2,
        dot_reductions: 4,
        norm_count: 5,
        axpy_passes: 6,
        basis_bytes_touched: 128,
        synchronization_points: 7,
        exact_residual_checks: 8,
    }
}

fn f32_solution() -> BlockFgmresSolution<f32> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, -1.0, 0.5]),
        iterations: 6,
        residuals: vec![1.0e-5, 2.0e-5],
        true_residuals: vec![3.0e-6, 4.0e-6],
        backward_errors: vec![5.0e-7, 6.0e-7],
        attainable_accuracy: vec![AccuracyStatus::Attained, AccuracyStatus::NotAttained],
        converged: false,
        op_counts: counts(),
    }
}

fn f64_solution() -> BlockFgmresSolution<f64> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 2, vec![1.0, -0.5]),
        iterations: 7,
        residuals: vec![1.0e-9, 2.0e-9],
        true_residuals: vec![3.0e-10, 4.0e-10],
        backward_errors: vec![5.0e-11, 6.0e-11],
        attainable_accuracy: vec![AccuracyStatus::Attained, AccuracyStatus::Attained],
        converged: true,
        op_counts: counts(),
    }
}

fn complex32_solution() -> BlockFgmresSolution<Complex32> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(
            1,
            2,
            vec![Complex32::new(1.0, -2.0), Complex32::new(0.25, 0.5)],
        ),
        iterations: 9,
        residuals: vec![1.0e-4, 2.0e-4],
        true_residuals: vec![3.0e-5, 4.0e-5],
        backward_errors: vec![5.0e-6, 6.0e-6],
        attainable_accuracy: vec![AccuracyStatus::NotChecked, AccuracyStatus::Attained],
        converged: true,
        op_counts: counts(),
    }
}

fn complex64_solution() -> BlockFgmresSolution<Complex64> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(
            2,
            1,
            vec![Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.25)],
        ),
        iterations: 11,
        residuals: vec![1.0e-8],
        true_residuals: vec![2.0e-9],
        backward_errors: vec![3.0e-10],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: counts(),
    }
}

fn gmres_solution() -> BlockGmresSolution<Complex64> {
    let fgmres = complex64_solution();
    BlockGmresSolution {
        x: fgmres.x,
        iterations: fgmres.iterations,
        residuals: fgmres.residuals,
        true_residuals: fgmres.true_residuals,
        backward_errors: fgmres.backward_errors,
        attainable_accuracy: fgmres.attainable_accuracy,
        converged: fgmres.converged,
        op_counts: fgmres.op_counts,
    }
}

#[test]
fn typed_checkpoint_round_trips_real_and_complex_scalar_families() {
    let f32_source = f32_solution();
    let f32_restored = BlockFgmresSolution::<f32>::from_typed_checkpoint_bytes(
        &f32_source
            .to_typed_checkpoint_bytes()
            .expect("f32 checkpoint encode"),
    )
    .expect("f32 checkpoint decode");
    assert_eq!(f32_restored.x.as_slice(), f32_source.x.as_slice());
    assert_eq!(f32_restored.residuals, f32_source.residuals);
    assert_eq!(f32_restored.op_counts, f32_source.op_counts);

    let f64_source = f64_solution();
    let f64_restored = BlockFgmresSolution::<f64>::from_typed_checkpoint_bytes(
        &f64_source
            .to_typed_checkpoint_bytes()
            .expect("f64 checkpoint encode"),
    )
    .expect("f64 checkpoint decode");
    assert_eq!(f64_restored.x.as_slice(), f64_source.x.as_slice());
    assert_eq!(f64_restored.residuals, f64_source.residuals);

    let complex32_source = complex32_solution();
    let complex32_restored = BlockFgmresSolution::<Complex32>::from_typed_checkpoint_bytes(
        &complex32_source
            .to_typed_checkpoint_bytes()
            .expect("Complex32 checkpoint encode"),
    )
    .expect("Complex32 checkpoint decode");
    assert_eq!(
        complex32_restored.x.as_slice(),
        complex32_source.x.as_slice()
    );
    assert_eq!(
        complex32_restored.attainable_accuracy,
        complex32_source.attainable_accuracy
    );

    let complex64_source = complex64_solution();
    let complex64_restored = BlockFgmresSolution::<Complex64>::from_typed_checkpoint_bytes(
        &complex64_source
            .to_typed_checkpoint_bytes()
            .expect("Complex64 checkpoint encode"),
    )
    .expect("Complex64 checkpoint decode");
    assert_eq!(
        complex64_restored.x.as_slice(),
        complex64_source.x.as_slice()
    );
    assert_eq!(
        complex64_restored.true_residuals,
        complex64_source.true_residuals
    );
}

#[test]
fn typed_checkpoint_rejects_cross_family_and_wire_corruption() {
    let source = f32_solution();
    let bytes = source
        .to_typed_checkpoint_bytes()
        .expect("f32 checkpoint encode");

    assert!(BlockFgmresSolution::<f64>::from_typed_checkpoint_bytes(&bytes).is_err());

    let mut truncated = bytes.clone();
    truncated.truncate(truncated.len() - 1);
    assert!(BlockFgmresSolution::<f32>::from_typed_checkpoint_bytes(&truncated).is_err());

    let mut trailing = bytes.clone();
    trailing.push(0xA5);
    assert!(
        BlockFgmresSolution::<f32>::from_typed_checkpoint_bytes(&trailing)
            .expect_err("trailing bytes")
            .to_string()
            .contains("trailing bytes")
    );

    let mut scalar_tag = bytes;
    scalar_tag[10] = 0xFF;
    assert!(BlockFgmresSolution::<f32>::from_typed_checkpoint_bytes(&scalar_tag).is_err());
}

#[test]
fn typed_checkpoint_rejects_non_finite_and_inconsistent_results() {
    let mut non_finite = f32_solution();
    non_finite.x.as_mut_slice()[0] = f32::NAN;
    assert!(non_finite.to_typed_checkpoint_bytes().is_err());

    let mut inconsistent = f32_solution();
    inconsistent.true_residuals.pop();
    assert!(inconsistent.to_typed_checkpoint_bytes().is_err());
}

#[test]
fn typed_checkpoint_round_trips_block_gmres_with_shared_wire_contract() {
    let source = gmres_solution();
    let bytes = source
        .to_typed_checkpoint_bytes()
        .expect("block GMRES checkpoint encode");
    let restored = BlockGmresSolution::<Complex64>::from_typed_checkpoint_bytes(&bytes)
        .expect("block GMRES checkpoint decode");
    assert_eq!(restored.x.as_slice(), source.x.as_slice());
    assert_eq!(restored.true_residuals, source.true_residuals);
    assert_eq!(restored.op_counts, source.op_counts);
}
