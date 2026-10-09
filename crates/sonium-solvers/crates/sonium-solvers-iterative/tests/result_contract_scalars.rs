use num_complex::{Complex32, Complex64};
use num_traits::{Float, Zero};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{
    AccuracyStatus, ComplexField, SolverResultContract, SolverStatus,
};
use sonium_solvers_iterative::{
    BlockFgmresSolution, BlockGmresSolution, GmresSolution, KrylovOpCounts,
};

fn scalar_contract_is_valid_for<T: ComplexField>() {
    let result = GmresSolution {
        x: vec![T::one()],
        iterations: 1,
        restarts: 0,
        residual: T::Real::zero(),
        estimated_residual: T::Real::zero(),
        backward_error: T::Real::zero(),
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::Converged,
        op_counts: KrylovOpCounts::default(),
    };

    assert_eq!(result.solver_status(), SolverStatus::Converged);
    assert_eq!(result.accuracy_status(), AccuracyStatus::Attained);
    result.validate_contract().unwrap();

    let block = BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![T::one()]),
        iterations: 1,
        residuals: vec![T::Real::zero()],
        true_residuals: vec![T::Real::zero()],
        backward_errors: vec![T::Real::zero()],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: true,
        op_counts: KrylovOpCounts::default(),
    };

    assert_eq!(block.solver_status(), SolverStatus::Converged);
    assert_eq!(block.accuracy_status(), AccuracyStatus::Attained);
    block.validate_contract().unwrap();
}

#[test]
fn result_contract_covers_real_and_complex_scalar_families() {
    scalar_contract_is_valid_for::<f32>();
    scalar_contract_is_valid_for::<f64>();
    scalar_contract_is_valid_for::<Complex32>();
    scalar_contract_is_valid_for::<Complex64>();
}

#[test]
fn numerical_error_sentinel_is_valid_for_all_scalar_families() {
    numerical_error_is_valid_for::<f32>();
    numerical_error_is_valid_for::<f64>();
    numerical_error_is_valid_for::<Complex32>();
    numerical_error_is_valid_for::<Complex64>();
}

#[test]
fn block_numerical_error_sentinel_is_valid_for_both_block_families() {
    let gmres = BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
        iterations: 0,
        residuals: vec![f64::INFINITY],
        true_residuals: vec![f64::INFINITY],
        backward_errors: vec![f64::INFINITY],
        attainable_accuracy: vec![AccuracyStatus::NotChecked],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    assert_eq!(gmres.solver_status(), SolverStatus::NumericalError);
    gmres.validate_contract().unwrap();

    let fgmres = BlockFgmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
        iterations: 0,
        residuals: vec![f64::INFINITY],
        true_residuals: vec![f64::INFINITY],
        backward_errors: vec![f64::INFINITY],
        attainable_accuracy: vec![AccuracyStatus::NotAttained],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    assert_eq!(fgmres.solver_status(), SolverStatus::NumericalError);
    fgmres.validate_contract().unwrap();
}

#[test]
fn scalar_contract_rejects_negative_diagnostics() {
    for field in 0..3 {
        let mut result = GmresSolution {
            x: vec![0.0_f64],
            iterations: 1,
            restarts: 0,
            residual: 1.0,
            estimated_residual: 1.0,
            backward_error: 1.0,
            attainable_accuracy: AccuracyStatus::NotAttained,
            converged: false,
            status: SolverStatus::MaxIterationsReached,
            op_counts: KrylovOpCounts::default(),
        };
        match field {
            0 => result.residual = -1.0,
            1 => result.estimated_residual = -1.0,
            _ => result.backward_error = -1.0,
        }
        assert!(
            result.validate_contract().is_err(),
            "field {field} accepted"
        );
    }
}

#[test]
fn block_contract_rejects_negative_diagnostics_for_both_families() {
    for field in 0..3 {
        let mut gmres = BlockGmresSolution {
            x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
            iterations: 1,
            residuals: vec![1.0],
            true_residuals: vec![1.0],
            backward_errors: vec![1.0],
            attainable_accuracy: vec![AccuracyStatus::NotAttained],
            converged: false,
            op_counts: KrylovOpCounts::default(),
        };
        let mut fgmres = BlockFgmresSolution {
            x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
            iterations: 1,
            residuals: vec![1.0],
            true_residuals: vec![1.0],
            backward_errors: vec![1.0],
            attainable_accuracy: vec![AccuracyStatus::NotAttained],
            converged: false,
            op_counts: KrylovOpCounts::default(),
        };
        match field {
            0 => {
                gmres.residuals[0] = -1.0;
                fgmres.residuals[0] = -1.0;
            }
            1 => {
                gmres.true_residuals[0] = -1.0;
                fgmres.true_residuals[0] = -1.0;
            }
            _ => {
                gmres.backward_errors[0] = -1.0;
                fgmres.backward_errors[0] = -1.0;
            }
        }
        assert!(
            gmres.validate_contract().is_err(),
            "block GMRES field {field} accepted"
        );
        assert!(
            fgmres.validate_contract().is_err(),
            "block FGMRES field {field} accepted"
        );
    }
}

#[test]
fn numerical_error_contract_rejects_contradictory_success_metadata() {
    let mut scalar = GmresSolution {
        x: vec![0.0_f64],
        iterations: 0,
        restarts: 0,
        residual: f64::INFINITY,
        estimated_residual: f64::INFINITY,
        backward_error: f64::INFINITY,
        attainable_accuracy: AccuracyStatus::Attained,
        converged: true,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    };
    assert!(scalar.validate_contract().is_err());

    scalar.converged = false;
    scalar.attainable_accuracy = AccuracyStatus::NotChecked;
    scalar.validate_contract().unwrap();

    let block = BlockGmresSolution {
        x: DenseMatrix::from_row_major_vec(1, 1, vec![0.0_f64]),
        iterations: 0,
        residuals: vec![f64::INFINITY],
        true_residuals: vec![f64::INFINITY],
        backward_errors: vec![f64::INFINITY],
        attainable_accuracy: vec![AccuracyStatus::Attained],
        converged: false,
        op_counts: KrylovOpCounts::default(),
    };
    assert!(block.validate_contract().is_err());
}

fn numerical_error_is_valid_for<T: ComplexField>() {
    let result = GmresSolution {
        x: vec![T::zero()],
        iterations: 0,
        restarts: 0,
        residual: T::Real::infinity(),
        estimated_residual: T::Real::infinity(),
        backward_error: T::Real::infinity(),
        attainable_accuracy: AccuracyStatus::NotChecked,
        converged: false,
        status: SolverStatus::NumericalError,
        op_counts: KrylovOpCounts::default(),
    };

    assert_eq!(result.solver_status(), SolverStatus::NumericalError);
    result.validate_contract().unwrap();
}
