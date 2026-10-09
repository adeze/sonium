use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::AccuracyStatus;
use sonium_solvers_iterative::{BlockFgmresSolution, KrylovOpCounts};

fn solution() -> BlockFgmresSolution<f64> {
    BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(2, 2, vec![1.0, 2.0, -1.0, 0.5]),
        iterations: 6,
        residuals: vec![1.0e-8, 2.0e-8],
        true_residuals: vec![3.0e-9, 4.0e-9],
        backward_errors: vec![5.0e-10, 6.0e-10],
        attainable_accuracy: vec![AccuracyStatus::Attained, AccuracyStatus::NotAttained],
        converged: false,
        op_counts: KrylovOpCounts {
            spmv_count: 3,
            preconditioner_count: 2,
            dot_reductions: 4,
            norm_count: 5,
            axpy_passes: 6,
            basis_bytes_touched: 128,
            synchronization_points: 7,
            exact_residual_checks: 8,
        },
    }
}

#[test]
fn native_block_checkpoint_round_trips_all_state() {
    let original = solution();
    let bytes = original.to_checkpoint_bytes().expect("encode");
    let restored = BlockFgmresSolution::from_checkpoint_bytes(&bytes).expect("decode");
    assert_eq!(restored.x.as_slice(), original.x.as_slice());
    assert_eq!(restored.iterations, original.iterations);
    assert_eq!(restored.residuals, original.residuals);
    assert_eq!(restored.true_residuals, original.true_residuals);
    assert_eq!(restored.backward_errors, original.backward_errors);
    assert_eq!(restored.attainable_accuracy, original.attainable_accuracy);
    assert_eq!(restored.op_counts, original.op_counts);
    assert!(!restored.converged);
}

#[test]
fn native_block_checkpoint_rejects_corruption() {
    let bytes = solution().to_checkpoint_bytes().expect("encode");

    let mut truncated = bytes.clone();
    truncated.truncate(truncated.len() - 1);
    assert!(BlockFgmresSolution::from_checkpoint_bytes(&truncated).is_err());

    let mut trailing = bytes.clone();
    trailing.push(0xA5);
    assert!(
        BlockFgmresSolution::from_checkpoint_bytes(&trailing)
            .expect_err("trailing bytes")
            .to_string()
            .contains("trailing bytes")
    );

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xFF;
    assert!(BlockFgmresSolution::from_checkpoint_bytes(&bad_magic).is_err());

    let mut bad_status = bytes;
    let status_offset = bad_status.len() - 8 * 8 - 2 - 1;
    bad_status[status_offset] = 0xFF;
    assert!(BlockFgmresSolution::from_checkpoint_bytes(&bad_status).is_err());
}

#[test]
fn native_block_checkpoint_rejects_invalid_state() {
    let mut invalid = solution();
    invalid.true_residuals.pop();
    assert!(invalid.to_checkpoint_bytes().is_err());

    let mut non_finite = solution();
    non_finite.backward_errors[0] = f64::NAN;
    assert!(non_finite.to_checkpoint_bytes().is_err());
}
