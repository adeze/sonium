//! Cross-family deterministic failure-exit matrix.

use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::differentiable::DifferentiableLinearSolve;
use sonium_solvers::direct::{
    DenseDirectConfig, DenseDirectMethod, MatrixClass, Ordering, SparseDirectConfig,
    SparseDirectMethod, factorize_dense, factorize_sparse,
};
use sonium_solvers::eigen::{LobpcgConfig, lobpcg};
use sonium_solvers::nonlinear::{NewtonConfig, newton_raphson};
use sonium_solvers::optimization::{Bound, SqpConfig, sqp};
use sonium_solvers::preconditioners::IluPreconditioner;
use sonium_solvers::time_integration::{NewmarkConfig, NewmarkInitial, try_newmark_integrate};
use sonium_solvers::{GmresConfig, IdentityPreconditioner};
use std::time::Instant;

const CASE_ID: &str = "solvers.failure-exit-matrix.v1";

fn sparse_malformed() -> CsrMatrix<f64> {
    let mut matrix = CsrMatrix::identity(2);
    matrix.row_ptrs[2] = 1;
    matrix
}

fn run() -> Result<Vec<(&'static str, bool)>, String> {
    let dense_nonfinite = factorize_dense(
        &DenseMatrix::from_row_major_vec(1, 1, vec![f64::NAN]),
        &DenseDirectConfig {
            matrix_class: MatrixClass::General,
            method: DenseDirectMethod::Lu,
            ..Default::default()
        },
    )
    .is_err();

    let sparse_noncanonical = factorize_sparse(
        &sparse_malformed(),
        &SparseDirectConfig {
            matrix_class: MatrixClass::General,
            method: SparseDirectMethod::SimplicialLu,
            ordering: Ordering::Auto,
            ..Default::default()
        },
        None,
    )
    .is_err();

    let ilu_zero_pivot = IluPreconditioner::try_from_csr(&CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0],
    ))
    .is_err();

    let newton_callback = newton_raphson(
        &[0.0_f64],
        |_| panic!("failure-exit matrix residual callback"),
        |_x, rhs| Ok(rhs.to_vec()),
        &NewtonConfig::default(),
    )
    .is_err();

    let sqp_invalid_bounds = sqp(
        &|x: &[f64]| x[0] * x[0],
        &|x: &[f64]| vec![2.0 * x[0]],
        &[0.0],
        &[],
        &SqpConfig {
            bounds: vec![Bound::both(1.0, 0.0)],
            ..SqpConfig::default()
        },
    )
    .is_err();

    let mut nan_operator = CsrMatrix::identity(2);
    nan_operator.values[0] = f64::NAN;
    let lobpcg_nonfinite = lobpcg(
        &nan_operator,
        &LobpcgConfig {
            num_eigenvalues: 1,
            ..LobpcgConfig::default()
        },
    )
    .is_err();

    let newmark_dimension = try_newmark_integrate(
        None::<&CsrMatrix<f64>>,
        &CsrMatrix::identity(2),
        |_| vec![0.0],
        &NewmarkInitial {
            displacement: vec![1.0],
            velocity: vec![0.0],
            acceleration: vec![-1.0],
        },
        |_rhs| vec![0.0],
        &NewmarkConfig::average_acceleration(0.1, 1),
    )
    .is_err();

    let differentiable_matrix = CsrMatrix::<f64>::identity(1);
    let differentiable_preconditioner = IdentityPreconditioner;
    let differentiable_config = DifferentiableLinearSolve::new(
        &differentiable_matrix,
        &differentiable_preconditioner,
        GmresConfig {
            tolerance: f64::NAN,
            ..GmresConfig::default()
        },
    )
    .is_err();

    Ok(vec![
        ("dense_nonfinite_matrix", dense_nonfinite),
        ("sparse_noncanonical_structure", sparse_noncanonical),
        ("ilu_zero_pivot", ilu_zero_pivot),
        ("newton_callback_panic", newton_callback),
        ("sqp_invalid_bounds", sqp_invalid_bounds),
        ("lobpcg_nonfinite_operator", lobpcg_nonfinite),
        ("newmark_dimension_mismatch", newmark_dimension),
        ("differentiable_invalid_tolerance", differentiable_config),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(cases) => {
            let failures = cases.iter().filter(|(_, passed)| !passed).count() as f64;
            let count = cases.len() as f64;
            let mut result = if failures == 0.0 {
                QaResult::passed(CASE_ID)
            } else {
                QaResult::failed(CASE_ID, format!("{failures} failure exits did not reject"))
            };
            result = result
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("failure_case_count", count)
                .metric("rejected_case_count", count - failures)
                .metric("unexpected_success_count", failures)
                .minimum("failure_case_count", 8.0)
                .minimum("rejected_case_count", 8.0)
                .maximum("unexpected_success_count", 0.0)
                .reference_checksum("deterministic-fixture:solver-failure-exits-v1")
                .dimension("direct, preconditioner, nonlinear, optimization, eigen, transient")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, passed) in cases {
                result = result.metric(name, if passed { 1.0 } else { 0.0 });
            }
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            if failures == 0.0 {
                println!("SUCCESS");
            } else {
                println!("FAILED");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("FAILURE MATRIX FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
