use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{
    AdditiveSchwarzPreconditioner, BlockJacobiPreconditioner, IlutConfig, SchwarzCoarseSetup,
};
use sonium_solvers_sparse::CsrMatrix;

#[test]
fn block_jacobi_runtime_diagnostics_cover_forward_and_adjoint_paths() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let preconditioner = BlockJacobiPreconditioner::from_csr(&matrix, 2, IlutConfig::default());
    let rhs = vec![1.0, -2.0, 3.0, -4.0];

    assert!(preconditioner.try_apply(&rhs).is_ok());
    let _ = preconditioner.apply_transpose(&rhs);
    let _ = preconditioner.apply_hermitian(&rhs);

    let diagnostics = preconditioner.runtime_diagnostics();
    assert_eq!(diagnostics.apply_calls, 1);
    assert_eq!(diagnostics.transpose_calls, 1);
    assert_eq!(diagnostics.hermitian_calls, 1);
    assert_eq!(diagnostics.failed_applications, 0);
    assert_eq!(diagnostics.non_finite_inputs, 0);
    assert_eq!(diagnostics.non_finite_outputs, 0);

    assert!(preconditioner.try_apply(&[f64::NAN; 4]).is_err());
    let diagnostics = preconditioner.runtime_diagnostics();
    assert_eq!(diagnostics.non_finite_inputs, 1);
    assert_eq!(diagnostics.failed_applications, 1);
}

#[test]
fn schwarz_runtime_diagnostics_report_coarse_and_adjoint_paths() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let partitions = vec![vec![0, 1], vec![2, 3]];
    let preconditioner =
        AdditiveSchwarzPreconditioner::from_csr_partitions(&matrix, &partitions, 1)
            .with_nicolaides_coarse_space(&matrix);
    let rhs = vec![1.0, -2.0, 3.0, -4.0];

    assert_eq!(preconditioner.coarse_setup(), SchwarzCoarseSetup::Attached);
    assert_eq!(preconditioner.coarse_dimension(), 2);
    assert!(preconditioner.try_apply(&rhs).is_ok());
    let _ = preconditioner.apply_transpose(&rhs);
    let _ = preconditioner.apply_hermitian(&rhs);

    let diagnostics = preconditioner.runtime_diagnostics();
    assert_eq!(diagnostics.apply_calls, 1);
    assert_eq!(diagnostics.transpose_calls, 1);
    assert_eq!(diagnostics.hermitian_calls, 1);
    assert_eq!(diagnostics.coarse_correction_calls, 3);
    assert_eq!(diagnostics.coarse_solve_failures, 0);
    assert_eq!(diagnostics.coarse_setup, SchwarzCoarseSetup::Attached);
    assert_eq!(diagnostics.coarse_dimension, 2);
    assert_eq!(diagnostics.failed_applications, 0);

    assert!(preconditioner.try_apply(&[f64::INFINITY; 4]).is_err());
    let diagnostics = preconditioner.runtime_diagnostics();
    assert_eq!(diagnostics.non_finite_inputs, 1);
    assert_eq!(diagnostics.failed_applications, 1);
}
