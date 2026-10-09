use num_complex::{Complex32, Complex64};
use num_traits::FromPrimitive;
use sonium_solvers::{
    AccuracyStatus, CsrMatrix, GmresBasisCheckpoint, GmresBasisRun, GmresConfig, GmresSolution,
    IdentityPreconditioner, KrylovOpCounts, LinearOperator, Preconditioner, SolverError,
    SolverResultContract, SolverStatus, try_gmres,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn matrix<T: sonium_solvers::ComplexField>() -> CsrMatrix<T> {
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            T::from_real(T::Real::from_f64(5.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(5.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(5.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(-1.0).unwrap()),
            T::from_real(T::Real::from_f64(5.0).unwrap()),
        ],
    )
}

#[test]
fn basis_capture_runs_implement_the_shared_result_contract() {
    let operator = matrix::<f64>();
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let checkpoint_run =
        GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config(), 2)
            .expect("fixture should capture an active basis");
    assert_eq!(
        checkpoint_run.solver_status(),
        SolverStatus::MaxIterationsReached
    );
    assert_eq!(
        checkpoint_run.accuracy_status(),
        AccuracyStatus::NotAttained
    );
    checkpoint_run
        .validate_contract()
        .expect("captured basis state should satisfy the shared contract");

    let solution_run = GmresBasisRun::Solution(GmresSolution {
        x: vec![1.0],
        iterations: 0,
        restarts: 0,
        residual: 0.0,
        estimated_residual: 0.0,
        backward_error: 0.0,
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::Converged,
        op_counts: KrylovOpCounts::default(),
    });
    assert_eq!(solution_run.solver_status(), SolverStatus::Converged);
    assert_eq!(solution_run.accuracy_status(), AccuracyStatus::Attained);
    solution_run
        .validate_contract()
        .expect("completed solution should delegate to GMRES contract");

    let invalid_run = GmresBasisRun::Solution(GmresSolution {
        x: vec![f64::NAN],
        iterations: 0,
        restarts: 0,
        residual: 0.0,
        estimated_residual: 0.0,
        backward_error: 0.0,
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::Converged,
        op_counts: KrylovOpCounts::default(),
    });
    assert!(invalid_run.validate_contract().is_err());
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 30,
        restart: 4,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    }
}

struct LateNanOperator {
    matrix: CsrMatrix<f64>,
    calls: Arc<AtomicUsize>,
}

impl LinearOperator<f64> for LateNanOperator {
    fn num_rows(&self) -> usize {
        self.matrix.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.matrix.num_cols()
    }

    fn apply_into(&self, input: &[f64], output: &mut [f64]) {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 4 {
            output[0] = f64::NAN;
            return;
        }
        self.matrix.apply_into(input, output);
    }

    fn apply_transpose_into(&self, input: &[f64], output: &mut [f64]) {
        self.matrix.apply_transpose_into(input, output);
    }
}

struct NanPreconditioner;

impl Preconditioner<f64> for NanPreconditioner {
    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }

    fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }
}

#[test]
fn basis_capture_preconditioner_failure_is_typed() {
    let operator = matrix::<f64>();
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let config = GmresConfig {
        max_iterations: 4,
        restart: 4,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };
    let error = GmresBasisCheckpoint::capture(&operator, &NanPreconditioner, &rhs, &config, 1)
        .expect_err("non-finite preconditioner output should be rejected");
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES basis replay preconditioner",
            ref reason,
        } if reason.contains("NaN or infinity")
    ));
}

#[test]
fn basis_replay_final_operator_failure_is_typed() {
    let calls = Arc::new(AtomicUsize::new(0));
    let operator = LateNanOperator {
        matrix: matrix::<f64>(),
        calls: Arc::clone(&calls),
    };
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let config = GmresConfig {
        max_iterations: 2,
        restart: 4,
        tolerance: 1.0e-12,
        ..GmresConfig::default()
    };
    let run = GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 1)
        .expect("first Arnoldi step should be checkpointed");
    let checkpoint = match run {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => panic!("fixture should require replay"),
    };
    let error = checkpoint
        .resume(&operator, &preconditioner, &rhs, &config)
        .expect_err("final residual callback failure should be reported");
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES basis replay operator",
            ref reason,
        } if reason.contains("NaN or infinity")
    ));
}

fn assert_scalar_basis_checkpoint_round_trip<T: sonium_solvers::ComplexField>()
where
    T::Real: FromPrimitive,
    GmresConfig<T::Real>: Default,
{
    let operator = matrix::<T>();
    let preconditioner = IdentityPreconditioner;
    let rhs = [
        T::from_real(T::Real::from_f64(1.0).unwrap()),
        T::from_real(T::Real::from_f64(2.0).unwrap()),
        T::from_real(T::Real::from_f64(-1.0).unwrap()),
        T::from_real(T::Real::from_f64(3.0).unwrap()),
    ];
    let tolerance = T::Real::from_f64(1.0e-5).unwrap();
    let config = GmresConfig {
        max_iterations: 30,
        restart: 4,
        tolerance,
        ..GmresConfig::default()
    };
    let run = GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 2)
        .expect("scalar fixture should capture an active basis");
    let checkpoint = match run {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => panic!("scalar fixture should require replay"),
    };
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .expect("scalar checkpoint should serialize");
    let restored = GmresBasisCheckpoint::<T>::from_checkpoint_bytes(&bytes)
        .expect("scalar checkpoint should restore with its scalar family");
    assert_eq!(restored.iterations(), checkpoint.iterations());
    assert_eq!(restored.basis_vectors(), checkpoint.basis_vectors());

    let resumed = restored
        .resume(&operator, &preconditioner, &rhs, &config)
        .expect("scalar checkpoint should resume");
    assert!(
        resumed.converged,
        "scalar replay residual: {:?}",
        resumed.residual
    );
    assert!(resumed.residual <= tolerance);

    let fresh = try_gmres(&operator, &rhs, &config).expect("fresh scalar solve should converge");
    for (actual, expected) in resumed.x.iter().zip(&fresh.x) {
        assert!((*actual - *expected).norm() < T::Real::from_f64(1.0e-3).unwrap());
    }
}

#[test]
fn basis_checkpoint_supports_all_scalar_family_round_trips() {
    assert_scalar_basis_checkpoint_round_trip::<f32>();
    assert_scalar_basis_checkpoint_round_trip::<Complex32>();
    assert_scalar_basis_checkpoint_round_trip::<Complex64>();
}

#[test]
fn basis_checkpoint_rejects_cross_family_scalar_restore() {
    let operator = matrix::<f32>();
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0_f32, 2.0, -1.0, 3.0];
    let config = GmresConfig {
        max_iterations: 30,
        restart: 4,
        tolerance: 1.0e-5_f32,
        ..GmresConfig::default()
    };
    let checkpoint =
        match GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 3)
            .expect("f32 fixture should capture")
        {
            GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
            GmresBasisRun::Solution(_) => panic!("f32 fixture should require replay"),
        };
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .expect("f32 checkpoint should serialize");
    assert!(GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bytes).is_err());

    let complex_operator = matrix::<Complex32>();
    let complex_rhs = [
        Complex32::new(1.0, 0.25),
        Complex32::new(2.0, -0.5),
        Complex32::new(-1.0, 0.75),
        Complex32::new(3.0, -0.25),
    ];
    let complex_checkpoint = match GmresBasisCheckpoint::capture(
        &complex_operator,
        &preconditioner,
        &complex_rhs,
        &config,
        3,
    )
    .expect("Complex32 fixture should capture")
    {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => panic!("Complex32 fixture should require replay"),
    };
    let complex_bytes = complex_checkpoint
        .to_checkpoint_bytes()
        .expect("Complex32 checkpoint should serialize");
    assert!(GmresBasisCheckpoint::<Complex64>::from_checkpoint_bytes(&complex_bytes).is_err());
}

#[test]
fn basis_checkpoint_supports_adaptive_restart_replay() {
    let operator = matrix::<f64>();
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let config = GmresConfig {
        max_iterations: 30,
        restart: 2,
        tolerance: 1.0e-12,
        min_restart: Some(1),
        max_restart: Some(4),
        ..GmresConfig::default()
    };
    let checkpoint =
        match GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 3)
            .expect("adaptive fixture should capture")
        {
            GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
            GmresBasisRun::Solution(_) => panic!("adaptive fixture should require replay"),
        };
    assert!((1..=4).contains(&checkpoint.restart_parameter()));
    assert_ne!(checkpoint.restart_parameter(), config.restart);
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .expect("adaptive checkpoint should serialize");
    let restored = GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bytes)
        .expect("adaptive checkpoint should restore");
    let resumed = restored
        .resume(&operator, &preconditioner, &rhs, &config)
        .expect("adaptive checkpoint should resume");
    assert!(
        resumed.converged,
        "adaptive replay residual: {:?}",
        resumed.residual
    );
    assert!(resumed.residual <= config.tolerance);
}

#[test]
fn basis_checkpoint_replays_active_arnoldi_state() {
    let operator = matrix::<f64>();
    let preconditioner = IdentityPreconditioner;
    let rhs = [1.0, 2.0, -1.0, 3.0];
    let run =
        GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config(), 2).unwrap();
    let checkpoint = match run {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => panic!("fixture should require replay"),
    };
    assert_eq!(checkpoint.iterations(), 2);
    assert!(checkpoint.basis_vectors() >= 3);
    assert!(checkpoint.storage_bytes() > 0);
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    assert!(bytes.len() > checkpoint.storage_bytes());
    let restored = GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.iterations(), checkpoint.iterations());
    assert_eq!(restored.basis_vectors(), checkpoint.basis_vectors());

    let resumed = restored
        .resume(&operator, &preconditioner, &rhs, &config())
        .unwrap();
    let fresh = try_gmres(&operator, &rhs, &config()).unwrap();
    assert!(resumed.converged);
    assert!(resumed.residual < 1.0e-10);
    for (actual, expected) in resumed.x.iter().zip(&fresh.x) {
        assert!((actual - expected).abs() < 1.0e-9);
    }
}

#[test]
fn complex_basis_checkpoint_replays_without_scalar_conversion() {
    let operator = matrix::<Complex64>();
    let preconditioner = IdentityPreconditioner;
    let rhs = [
        Complex64::new(1.0, 0.25),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 0.75),
        Complex64::new(3.0, -0.25),
    ];
    let checkpoint = match GmresBasisCheckpoint::capture(
        &operator,
        &preconditioner,
        &rhs,
        &GmresConfig {
            max_iterations: 30,
            restart: 4,
            tolerance: 1.0e-11,
            ..GmresConfig::default()
        },
        1,
    )
    .unwrap()
    {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => panic!("fixture should require replay"),
    };
    let resumed = checkpoint
        .resume(
            &operator,
            &preconditioner,
            &rhs,
            &GmresConfig {
                max_iterations: 30,
                restart: 4,
                tolerance: 1.0e-11,
                ..GmresConfig::default()
            },
        )
        .unwrap();
    assert!(resumed.converged);
    assert!(resumed.residual < 1.0e-9);
}

#[test]
fn basis_checkpoint_bytes_reject_corruption_and_scalar_mismatch() {
    let operator = matrix::<f64>();
    let preconditioner = IdentityPreconditioner;
    let config = config();
    let checkpoint = match GmresBasisCheckpoint::capture(
        &operator,
        &preconditioner,
        &[1.0, 2.0, -1.0, 3.0],
        &config,
        2,
    )
    .unwrap()
    {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => panic!("fixture should require replay"),
    };
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bad_magic).is_err());
    assert!(GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&trailing).is_err());
    let mut bad_version = bytes.clone();
    bad_version[8] = 2;
    assert!(GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bad_version).is_err());
    let mut bad_flags = bytes.clone();
    bad_flags[13] = 1;
    assert!(GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&bad_flags).is_err());
    assert!(GmresBasisCheckpoint::<Complex64>::from_checkpoint_bytes(&bytes).is_err());
}

#[test]
fn basis_checkpoint_rejects_unsafe_requests_before_operator_work() {
    let operator = CsrMatrix::<f64>::identity(2);
    let preconditioner = IdentityPreconditioner;
    let config = GmresConfig {
        max_iterations: 4,
        restart: 2,
        tolerance: 1.0e-8,
        ..GmresConfig::default()
    };
    let error = GmresBasisCheckpoint::capture(&operator, &preconditioner, &[1.0, 2.0], &config, 0)
        .expect_err("zero checkpoint budget must be rejected");
    assert!(matches!(error, SolverError::OperationFailed { .. }));
    assert!(
        GmresBasisCheckpoint::capture(&operator, &preconditioner, &[1.0, f64::NAN], &config, 1,)
            .is_err()
    );
}
