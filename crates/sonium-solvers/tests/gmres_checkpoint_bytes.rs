use num_complex::Complex64;
use sonium_solvers::SolverResultContract;
use sonium_solvers::{CsrMatrix, GmresCheckpoint, GmresConfig, SolverError, try_gmres};

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ],
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

#[test]
fn portable_checkpoint_round_trip_preserves_solution_metadata_and_resume() {
    let operator = matrix();
    let first = try_gmres(&operator, &[5.0, 0.0, 3.0], &config()).unwrap();
    let checkpoint = GmresCheckpoint::from_solution(&first).unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    assert!(bytes.len() > checkpoint.storage_bytes());

    let restored = GmresCheckpoint::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.solution(), checkpoint.solution());
    assert_eq!(restored.dimension(), 3);
    assert_eq!(restored.source_iterations(), checkpoint.source_iterations());
    assert_eq!(restored.source_restarts(), checkpoint.source_restarts());
    assert_eq!(restored.source_residual(), checkpoint.source_residual());

    let resumed = restored
        .resume(&operator, &[1.0, 7.0, 6.0], &config())
        .unwrap();
    assert!(resumed.converged);
    assert!(resumed.residual < 1.0e-10);
    resumed.validate_contract().unwrap();
}

#[test]
fn portable_complex_checkpoint_round_trip_preserves_values() {
    let values = [Complex64::new(1.25, -0.5), Complex64::new(-2.0, 3.75)];
    let checkpoint = GmresCheckpoint::from_iterate(&values).unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored = GmresCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.solution(), &values);
    assert_eq!(restored.source_iterations(), 0);
    assert_eq!(restored.source_restarts(), 0);
    assert_eq!(restored.source_residual(), 0.0);
}

#[test]
fn portable_checkpoint_rejects_corruption_and_malformed_metadata() {
    let checkpoint = GmresCheckpoint::from_iterate(&[0.0_f64, 1.0, -2.0]).unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&trailing).is_err());

    let mut bad_version = bytes.clone();
    bad_version[8] = 2;
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&bad_version).is_err());
    assert!(GmresCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes).is_err());

    let mut bad_flags = bytes.clone();
    bad_flags[13] = 1;
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&bad_flags).is_err());

    let mut bad_dimension = bytes.clone();
    bad_dimension[14..22].copy_from_slice(&2_u64.to_le_bytes());
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&bad_dimension).is_err());

    let mut bad_residual = bytes.clone();
    bad_residual[38..46].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&bad_residual).is_err());

    let mut huge_solution = bytes.clone();
    huge_solution[46..54].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(GmresCheckpoint::<f64>::from_checkpoint_bytes(&huge_solution).is_err());

    let invalid = sonium_solvers::GmresSolution {
        x: vec![0.0_f64],
        iterations: 1,
        restarts: 0,
        residual: f64::NAN,
        estimated_residual: 0.0,
        backward_error: 0.0,
        attainable_accuracy: sonium_solvers::AccuracyStatus::NotChecked,
        converged: false,
        status: sonium_solvers::SolverStatus::MaxIterationsReached,
        op_counts: Default::default(),
    };
    let error = GmresCheckpoint::from_solution(&invalid)
        .expect_err("non-finite residual must not be captured");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
}
