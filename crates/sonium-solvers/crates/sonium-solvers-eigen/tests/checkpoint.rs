use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_eigen::{
    ComplexTargetedEigenResult, EIGEN_CHECKPOINT_FORMAT_VERSION, LobpcgResult, TargetedEigenResult,
};

fn real_matrix() -> DenseMatrix<f64> {
    DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.25])
}

fn complex_matrix() -> DenseMatrix<Complex64> {
    DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(2.0, 0.0),
        ],
    )
}

#[test]
fn real_targeted_and_lobpcg_checkpoints_round_trip() {
    let targeted = TargetedEigenResult {
        eigenvalues: vec![1.25, 2.5],
        eigenvectors: real_matrix(),
        residual_norms: vec![1.0e-8, 2.0e-8],
        iterations: 9,
        converged: true,
    };
    let encoded = targeted.to_checkpoint_bytes().expect("targeted encode");
    let restored = TargetedEigenResult::from_checkpoint_bytes(&encoded).expect("targeted decode");
    assert_eq!(restored.eigenvalues, targeted.eigenvalues);
    assert_eq!(
        restored.eigenvectors.as_slice(),
        targeted.eigenvectors.as_slice()
    );
    assert_eq!(restored.residual_norms, targeted.residual_norms);
    assert_eq!(restored.iterations, targeted.iterations);
    assert_eq!(restored.converged, targeted.converged);

    let lobpcg = LobpcgResult {
        eigenvalues: targeted.eigenvalues.clone(),
        eigenvectors: targeted.eigenvectors.clone(),
        iterations: 4,
        residual_norms: targeted.residual_norms.clone(),
        converged: false,
    };
    let bytes = lobpcg.to_checkpoint_bytes().expect("LOBPCG encode");
    let restored = LobpcgResult::from_checkpoint_bytes(&bytes).expect("LOBPCG decode");
    assert_eq!(restored.eigenvalues, lobpcg.eigenvalues);
    assert_eq!(
        restored.eigenvectors.as_slice(),
        lobpcg.eigenvectors.as_slice()
    );
    assert_eq!(restored.iterations, lobpcg.iterations);
    assert!(!restored.converged);
}

#[test]
fn complex_targeted_checkpoint_round_trips() {
    let original = ComplexTargetedEigenResult {
        eigenvalues: vec![Complex64::new(1.0, -0.5), Complex64::new(2.0, 0.25)],
        eigenvectors: complex_matrix(),
        residual_norms: vec![1.0e-7, 2.0e-7],
        iterations: 11,
        converged: false,
    };
    let encoded = original.to_checkpoint_bytes().expect("complex encode");
    let restored =
        ComplexTargetedEigenResult::from_checkpoint_bytes(&encoded).expect("complex decode");
    assert_eq!(restored.eigenvalues, original.eigenvalues);
    assert_eq!(
        restored.eigenvectors.as_slice(),
        original.eigenvectors.as_slice()
    );
    assert_eq!(restored.residual_norms, original.residual_norms);
    assert_eq!(restored.iterations, original.iterations);
    assert!(!restored.converged);
}

#[test]
fn eigen_checkpoints_reject_kind_version_truncation_and_trailing_bytes() {
    let original = TargetedEigenResult {
        eigenvalues: vec![1.0, 2.0],
        eigenvectors: real_matrix(),
        residual_norms: vec![1.0e-8, 1.0e-8],
        iterations: 1,
        converged: true,
    };
    let encoded = original.to_checkpoint_bytes().expect("encode");

    let mut truncated = encoded.clone();
    truncated.truncate(truncated.len() - 1);
    assert!(TargetedEigenResult::from_checkpoint_bytes(&truncated).is_err());

    let mut trailing = encoded.clone();
    trailing.push(0xA5);
    assert!(
        TargetedEigenResult::from_checkpoint_bytes(&trailing)
            .expect_err("trailing bytes")
            .to_string()
            .contains("trailing bytes")
    );

    let mut bad_version = encoded.clone();
    bad_version[8..10].copy_from_slice(&(EIGEN_CHECKPOINT_FORMAT_VERSION + 1).to_le_bytes());
    assert!(
        TargetedEigenResult::from_checkpoint_bytes(&bad_version)
            .expect_err("version")
            .to_string()
            .contains("unsupported checkpoint version")
    );

    let mut wrong_kind = encoded;
    wrong_kind[10] = 3;
    assert!(
        TargetedEigenResult::from_checkpoint_bytes(&wrong_kind)
            .expect_err("kind")
            .to_string()
            .contains("kind mismatch")
    );
}

#[test]
fn eigen_checkpoints_reject_non_finite_and_shape_mismatch() {
    let mut non_finite = TargetedEigenResult {
        eigenvalues: vec![f64::NAN],
        eigenvectors: DenseMatrix::from_row_major_vec(1, 1, vec![1.0]),
        residual_norms: vec![1.0],
        iterations: 0,
        converged: false,
    };
    assert!(non_finite.to_checkpoint_bytes().is_err());

    non_finite.eigenvalues[0] = 1.0;
    non_finite.residual_norms.clear();
    assert!(non_finite.to_checkpoint_bytes().is_err());
}
