use num_complex::Complex64;
use sonium_solvers::{
    AccuracyStatus, ClosureDifferentiableSolve, ContractValidation, DifferentiableContractValue,
    DifferentiableSolveContract, DifferentiableSolveResult, DifferentiableSystemKind,
    GmresSolution, KrylovOpCounts, SolverResultContract, SolverStatus,
    eigen::{ComplexTargetedEigenResult, LobpcgResult, TargetedEigenResult},
};
use sonium_solvers_core::dense::DenseMatrix;

fn real_eigen_result() -> TargetedEigenResult {
    TargetedEigenResult {
        eigenvalues: vec![1.0, 2.0],
        eigenvectors: DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        residual_norms: vec![1.0e-10, 2.0e-10],
        iterations: 2,
        converged: true,
    }
}

fn complex_eigen_result() -> ComplexTargetedEigenResult {
    ComplexTargetedEigenResult {
        eigenvalues: vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)],
        eigenvectors: DenseMatrix::from_row_major_vec(
            3,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ),
        residual_norms: vec![1.0e-10, 2.0e-10],
        iterations: 2,
        converged: true,
    }
}

fn lobpcg_result() -> LobpcgResult {
    LobpcgResult {
        eigenvalues: vec![1.0, 2.0],
        eigenvectors: DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        iterations: 2,
        residual_norms: vec![1.0e-10, 2.0e-10],
        converged: true,
    }
}

fn gmres_result(status: SolverStatus, accuracy: AccuracyStatus) -> GmresSolution<f64> {
    GmresSolution {
        x: vec![1.0],
        iterations: 1,
        restarts: 0,
        residual: 0.0,
        estimated_residual: 0.0,
        backward_error: 0.0,
        attainable_accuracy: accuracy,
        converged: status == SolverStatus::Converged,
        status,
        op_counts: KrylovOpCounts::default(),
    }
}

#[test]
fn differentiable_result_contract_covers_forward_and_adjoint_solves() {
    let result = DifferentiableSolveResult {
        forward: gmres_result(SolverStatus::Converged, AccuracyStatus::Attained),
        adjoint: Some(gmres_result(
            SolverStatus::Converged,
            AccuracyStatus::Attained,
        )),
    };
    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    result.validate_contract().expect("finite solves are valid");
}

#[test]
fn differentiable_result_contract_rejects_bad_inner_result() {
    let result = DifferentiableSolveResult {
        forward: gmres_result(SolverStatus::Converged, AccuracyStatus::Attained),
        adjoint: Some(gmres_result(
            SolverStatus::Converged,
            AccuracyStatus::NotAttained,
        )),
    };
    assert!(result.validate_contract().is_err());
}

#[test]
fn differentiable_result_contract_rejects_forward_adjoint_dimension_mismatch() {
    let result = DifferentiableSolveResult {
        forward: gmres_result(SolverStatus::Converged, AccuracyStatus::Attained),
        adjoint: Some(GmresSolution {
            x: vec![1.0, 2.0],
            iterations: 1,
            restarts: 0,
            residual: 0.0,
            estimated_residual: 0.0,
            backward_error: 0.0,
            attainable_accuracy: AccuracyStatus::Attained,
            converged: true,
            status: SolverStatus::Converged,
            op_counts: KrylovOpCounts::default(),
        }),
    };

    let error = result
        .validate_contract()
        .expect_err("forward and adjoint dimensions must agree");
    assert!(matches!(
        error,
        sonium_solvers_core::traits::SolverError::DimensionMismatch { .. }
    ));
}

#[test]
fn native_eigen_results_are_differentiable_contract_values() {
    let real = real_eigen_result();
    assert_eq!(real.contract_dimension(), Some(3));
    real.validate_contract_value("real eigen forward")
        .expect("finite real eigen result is valid");

    let complex = complex_eigen_result();
    assert_eq!(complex.contract_dimension(), Some(3));
    complex
        .validate_contract_value("complex eigen forward")
        .expect("finite complex eigen result is valid");

    let lobpcg = lobpcg_result();
    assert_eq!(lobpcg.contract_dimension(), Some(3));
    lobpcg
        .validate_contract_value("LOBPCG forward")
        .expect("finite LOBPCG result is valid");
}

#[test]
fn native_eigen_contract_rejects_malformed_result_before_adjoint_callback() {
    let malformed = TargetedEigenResult {
        eigenvalues: vec![1.0, 2.0],
        eigenvectors: DenseMatrix::from_row_major_vec(3, 1, vec![1.0, 0.0, 0.0]),
        residual_norms: vec![1.0e-10, 2.0e-10],
        iterations: 1,
        converged: true,
    };
    assert!(
        malformed
            .validate_contract_value("malformed eigen")
            .is_err()
    );

    let input = vec![0.0_f64; 3];
    let contract = ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Eigen,
        |_input: &[f64]| Ok::<_, sonium_solvers::SolverError>(malformed.clone()),
        |_forward: &TargetedEigenResult, _seed: &[f64]| -> Result<TargetedEigenResult, sonium_solvers::SolverError> {
            panic!("adjoint callback must not run for malformed native state")
        },
        ContractValidation::<[f64], TargetedEigenResult, [f64], TargetedEigenResult>::
            dimension_checked(),
    );
    assert!(contract.solve_forward(&input).is_err());
}

#[test]
fn native_eigen_contract_supports_deterministic_replay() {
    let contract = ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Eigen,
        |_input: &[f64]| Ok::<_, sonium_solvers::SolverError>(real_eigen_result()),
        |_forward: &TargetedEigenResult, _seed: &[f64]| {
            Ok::<_, sonium_solvers::SolverError>(real_eigen_result())
        },
        ContractValidation::<[f64], TargetedEigenResult, [f64], TargetedEigenResult>::
            dimension_checked(),
    );
    let input = vec![0.0_f64; 3];
    let state = contract
        .solve_forward(&input)
        .expect("native eigen forward state");
    contract
        .solve_adjoint(&state, &input)
        .expect("native eigen adjoint state");
    let replay = sonium_solvers::differentiable_contract::verify_forward_replay(
        &contract,
        &input,
        |left: &TargetedEigenResult, right: &TargetedEigenResult| {
            left.eigenvalues == right.eigenvalues
                && left.eigenvectors.as_slice() == right.eigenvectors.as_slice()
                && left.residual_norms == right.residual_norms
        },
    )
    .expect("native eigen replay");
    assert!(replay.deterministic);
    assert_eq!(replay.evaluations, 2);
}
