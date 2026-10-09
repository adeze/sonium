use sonium_solvers_core::traits::{AccuracyStatus, SolverResultContract, SolverStatus};
use sonium_solvers_direct::{
    IrResult, SparseRefinementProvenance, SparseRefinementResult, SparseRefinementStopReason,
};

#[test]
fn iterative_refinement_results_implement_shared_result_contract() {
    let ir = IrResult {
        x: vec![1.0_f64],
        iterations: 1,
        final_residual: 1.0e-12,
        converged: true,
    };
    assert_eq!(ir.solver_status(), SolverStatus::Converged);
    assert_eq!(ir.accuracy_status(), AccuracyStatus::Attained);
    ir.validate_contract().expect("finite IR result is valid");

    let sparse = SparseRefinementResult {
        x: vec![1.0_f64],
        iterations: 1,
        initial_relative_residual: 1.0,
        relative_residual: 1.0e-12,
        tolerance: 1.0e-10,
        converged: true,
        stop_reason: SparseRefinementStopReason::ToleranceReached,
        provenance: SparseRefinementProvenance::ReusedValueOnlyPattern {
            structure_fingerprint: 7,
        },
    };
    assert_eq!(sparse.solver_status(), SolverStatus::Converged);
    sparse
        .validate_contract()
        .expect("finite sparse-refinement result is valid");
}

#[test]
fn iterative_refinement_contract_rejects_inconsistent_metadata() {
    let result = IrResult {
        x: vec![1.0_f64],
        iterations: 1,
        final_residual: -1.0,
        converged: true,
    };
    assert!(result.validate_contract().is_err());
}
