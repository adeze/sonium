use num_complex::Complex64;
use sonium_solvers::{
    AssumptionProvenance, CsrMatrix, LinearSolveConfig, LinearSolverType, LinearSystemAssumption,
    solve_complex, solve_real,
};

#[test]
fn small_spd_result_records_independent_validation() {
    let matrix = CsrMatrix::identity(2);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };

    let result = solve_real(&matrix, &[1.0, 2.0], &config).expect("identity CG solve");
    assert_eq!(result.assumption, LinearSystemAssumption::General);
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::IndependentlyValidated
    );
}

#[test]
fn complex_general_result_does_not_claim_spd_evidence() {
    let matrix = CsrMatrix::identity(2);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Gmres,
        max_iterations: 4,
        tolerance: 1e-12,
        ..Default::default()
    };
    let rhs = [Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0)];

    let result = solve_complex(&matrix, &rhs, &config).expect("identity GMRES solve");
    assert_eq!(result.assumption, LinearSystemAssumption::General);
    assert_eq!(
        result.assumption_provenance,
        AssumptionProvenance::NotRequired
    );
}
