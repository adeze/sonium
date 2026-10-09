use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers::SolverResultContract;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{
    ComplexField, DeflatedCheckpoint, DeflationSubspace, DiagonalPreconditioner, GmresConfig,
    SolverError, try_gmres_deflated, try_gmres_deflated_preconditioned,
};

fn scalar<T: ComplexField>(value: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_real(T::Real::from_f64(value).expect("finite scalar fixture"))
}

fn scalar_matrix<T: ComplexField>() -> CsrMatrix<T>
where
    T::Real: FromPrimitive,
{
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![scalar(4.0), scalar(3.0), scalar(2.0)],
    )
}

fn scalar_config<T: ComplexField>() -> GmresConfig<T::Real>
where
    T::Real: FromPrimitive,
    GmresConfig<T::Real>: Default,
{
    GmresConfig {
        max_iterations: 20,
        restart: 3,
        tolerance: T::Real::from_f64(1.0e-5).expect("finite tolerance"),
        print_interval: 0,
        ..Default::default()
    }
}

fn scalar_checkpoint_case<T: ComplexField>() -> (f64, f64, f64)
where
    T::Real: FromPrimitive + ToPrimitive,
    GmresConfig<T::Real>: Default,
{
    let operator = scalar_matrix::<T>();
    let deflation = DeflationSubspace::try_new(
        vec![vec![scalar(0.7), scalar(-0.3), scalar(0.5)]],
        &operator,
    )
    .expect("deflation fixture");
    let checkpoint = DeflatedCheckpoint::from_subspace(&deflation).expect("checkpoint capture");
    let bytes = checkpoint.to_checkpoint_bytes().expect("checkpoint encode");
    let restored =
        DeflatedCheckpoint::<T>::from_checkpoint_bytes(&bytes).expect("checkpoint decode");
    let vector_delta = restored
        .vectors()
        .iter()
        .zip(checkpoint.vectors())
        .flat_map(|(restored, original)| restored.iter().zip(original))
        .map(|(left, right)| {
            (*left - *right)
                .norm()
                .to_f64()
                .expect("finite vector delta")
        })
        .fold(0.0, f64::max);

    let rhs = vec![scalar(1.0), scalar(-2.0), scalar(3.0)];
    let fresh = try_gmres_deflated(&operator, &deflation, &rhs, None, &scalar_config::<T>())
        .expect("fresh deflated solve");
    let resumed = restored
        .resume(&operator, &rhs, None, &scalar_config::<T>())
        .expect("restored deflated solve");
    assert!(fresh.converged && resumed.converged);
    fresh.validate_contract().expect("fresh result contract");
    resumed
        .validate_contract()
        .expect("restored result contract");
    let solution_delta = fresh
        .x
        .iter()
        .zip(&resumed.x)
        .map(|(left, right)| (*left - *right).norm().to_f64().expect("finite delta"))
        .fold(0.0, f64::max);
    let residual = resumed.residual.to_f64().expect("finite residual");
    (
        solution_delta.max(vector_delta),
        residual,
        bytes.len() as f64,
    )
}

#[test]
fn deflated_checkpoint_round_trips_supported_scalar_families() {
    for (label, (solution_delta, residual, bytes)) in [
        ("f32", scalar_checkpoint_case::<f32>()),
        ("Complex32", scalar_checkpoint_case::<Complex32>()),
        ("Complex64", scalar_checkpoint_case::<Complex64>()),
    ] {
        assert!(
            solution_delta <= 1.0e-5,
            "{label} solution delta {solution_delta}"
        );
        assert!(residual <= 1.0e-5, "{label} residual {residual}");
        assert!(bytes > 0.0, "{label} checkpoint was empty");
    }
}

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_dense(
        &sonium_solvers::array![[4.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        1.0e-15,
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 20,
        restart: 3,
        tolerance: 1.0e-12,
        print_interval: 0,
        ..Default::default()
    }
}

#[test]
fn deflated_checkpoint_round_trip_resumes_against_current_operator() {
    let operator = matrix();
    let deflation = DeflationSubspace::try_new(vec![vec![1.0, 0.0, 0.0]], &operator).unwrap();
    let checkpoint = DeflatedCheckpoint::from_subspace(&deflation).unwrap();
    assert_eq!(checkpoint.dimension(), 3);
    assert_eq!(checkpoint.num_vectors(), 1);
    assert_eq!(checkpoint.storage_bytes(), 3 * std::mem::size_of::<f64>());

    let restored = DeflatedCheckpoint::<f64>::from_checkpoint_bytes(
        &checkpoint.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let rhs = [1.0, -2.0, 3.0];
    let fresh = try_gmres_deflated(&operator, &deflation, &rhs, None, &config()).unwrap();
    let resumed = restored.resume(&operator, &rhs, None, &config()).unwrap();
    assert!(fresh.converged && resumed.converged);
    resumed.validate_contract().unwrap();
    assert!(fresh.residual < 1.0e-12 && resumed.residual < 1.0e-12);
    for (left, right) in fresh.x.iter().zip(&resumed.x) {
        assert!((*left - *right).abs() < 1.0e-12);
    }

    let mut rebuilt = DeflationSubspace::try_new(Vec::new(), &operator).unwrap();
    restored.restore_into(&mut rebuilt, &operator).unwrap();
    assert_eq!(rebuilt.vectors(), restored.vectors());
}

#[test]
fn deflated_checkpoint_preconditioned_resume_matches_fresh_and_accepts_guess() {
    let operator = matrix();
    let preconditioner = DiagonalPreconditioner::from_csr(&operator);
    let deflation = DeflationSubspace::try_new(vec![vec![1.0, 0.0, 0.0]], &operator).unwrap();
    let checkpoint = DeflatedCheckpoint::from_subspace(&deflation).unwrap();
    let restored = DeflatedCheckpoint::<f64>::from_checkpoint_bytes(
        &checkpoint.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    let rhs = [1.0, -2.0, 3.0];
    let expected = [0.25, -2.0 / 3.0, 1.5];

    let fresh = try_gmres_deflated_preconditioned(
        &operator,
        &preconditioner,
        &deflation,
        &rhs,
        None,
        &config(),
    )
    .unwrap();
    let resumed = restored
        .resume_preconditioned(&operator, &preconditioner, &rhs, Some(&expected), &config())
        .unwrap();

    assert!(fresh.converged && resumed.converged);
    assert!(fresh.residual < 1.0e-12 && resumed.residual < 1.0e-12);
    resumed.validate_contract().unwrap();
    assert_eq!(resumed.iterations, 0);
    for (actual, reference) in fresh.x.iter().zip(expected) {
        assert!((*actual - reference).abs() < 1.0e-12);
    }
    assert!(
        restored
            .resume_preconditioned(
                &operator,
                &preconditioner,
                &rhs,
                Some(&[0.0, 0.0]),
                &config()
            )
            .is_err()
    );
}

#[test]
fn deflated_checkpoint_preserves_complex_vectors_and_rejects_malformed_data() {
    let vectors = vec![
        vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 0.1),
            Complex64::new(0.25, -0.75),
        ],
        vec![
            Complex64::new(0.1, -0.4),
            Complex64::new(0.75, 0.2),
            Complex64::new(-0.2, 0.6),
        ],
    ];
    let checkpoint = DeflatedCheckpoint::from_vectors(&vectors).unwrap();
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored = DeflatedCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.vectors(), vectors.as_slice());

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(matches!(
        DeflatedCheckpoint::<Complex64>::from_checkpoint_bytes(&bad_magic),
        Err(SolverError::OperationFailed { .. })
    ));
    assert!(
        DeflatedCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err()
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(DeflatedCheckpoint::<Complex64>::from_checkpoint_bytes(&trailing).is_err());
    assert!(DeflatedCheckpoint::<f64>::from_vectors(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    assert!(DeflatedCheckpoint::<f64>::from_vectors(&[vec![f64::NAN]]).is_err());

    let empty = DeflatedCheckpoint::<f64>::from_vectors(&[]).unwrap();
    let mut bad_empty_dimension = empty.to_checkpoint_bytes().unwrap();
    bad_empty_dimension[14..22].copy_from_slice(&1_u64.to_le_bytes());
    assert!(DeflatedCheckpoint::<f64>::from_checkpoint_bytes(&bad_empty_dimension).is_err());
}
