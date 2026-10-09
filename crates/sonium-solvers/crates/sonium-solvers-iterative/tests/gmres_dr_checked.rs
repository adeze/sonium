use sonium_solvers_core::traits::IdentityPreconditioner;
use sonium_solvers_core::traits::{LinearOperator, SolverError};
use sonium_solvers_iterative::{GmresDrConfig, try_gmres_dr_preconditioned};

struct PanicOperator;

struct NonFiniteOperator;

impl LinearOperator<f64> for PanicOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], _output: &mut [f64]) {
        panic!("GMRES-DR operator callback failed");
    }

    fn apply_transpose_into(&self, _input: &[f64], _output: &mut [f64]) {
        panic!("GMRES-DR transpose callback failed");
    }
}

impl LinearOperator<f64> for NonFiniteOperator {
    fn num_rows(&self) -> usize {
        1
    }

    fn num_cols(&self) -> usize {
        1
    }

    fn apply_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }

    fn apply_transpose_into(&self, _input: &[f64], output: &mut [f64]) {
        output[0] = f64::NAN;
    }
}

fn config() -> GmresDrConfig<f64> {
    GmresDrConfig {
        gmres: sonium_solvers_iterative::GmresConfig {
            max_iterations: 8,
            restart: 2,
            tolerance: 1.0e-12,
            ..sonium_solvers_iterative::GmresConfig::default()
        },
        num_deflation_vectors: 1,
    }
}

#[test]
fn gmres_dr_contains_operator_panics_as_typed_errors() {
    let error = try_gmres_dr_preconditioned(
        &PanicOperator,
        &IdentityPreconditioner,
        &[1.0],
        None,
        &config(),
    )
    .expect_err("operator panic must not escape the checked GMRES-DR boundary");

    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES-DR",
            ..
        }
    ));
    assert!(error.to_string().contains("operator callback failed"));
}

#[test]
fn gmres_dr_rejects_non_finite_operator_output() {
    let error = try_gmres_dr_preconditioned(
        &NonFiniteOperator,
        &IdentityPreconditioner,
        &[1.0],
        None,
        &config(),
    )
    .expect_err("non-finite operator output must be rejected");

    assert!(matches!(
        error,
        SolverError::OperationFailed {
            operation: "GMRES-DR",
            ..
        }
    ));
    assert!(error.to_string().contains("generated NaN or infinity"));
}
