//! Independent randomized CPU evidence for `LinearOperator` adjoint contracts.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_core::{
    dense::DenseMatrix,
    traits::{AdjointCapability, LinearOperator, SolverError},
};
use sonium_solvers_sparse::{
    CompactCsrMatrix, CsrMatrix, DenseBlockOperator, FnOperator, RowBucketedCsrMatrix,
    SellCSigmaMatrix,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.operator-adjoint-randomized.v1";
const TOLERANCE: f64 = 1.0e-10;

#[derive(Default)]
struct Metrics {
    max_hermitian_error: f64,
    max_transpose_error: f64,
    identity_checks: usize,
    block_checks: usize,
    rejection_checks: usize,
}

fn fixture() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        3,
        2,
        vec![0, 2, 3, 5],
        vec![0, 1, 1, 0, 1],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.25, -3.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-0.75, 2.5),
        ],
    )
}

fn property_vector(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let real = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let imag = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            Complex64::new(real, imag)
        })
        .collect()
}

fn check_block_parity(
    name: &str,
    operator: &dyn LinearOperator<Complex64>,
    metrics: &mut Metrics,
) -> Result<(), String> {
    let x = DenseMatrix::from_row_major_vec(
        operator.num_cols(),
        2,
        (0..operator.num_cols() * 2)
            .map(|index| Complex64::new((index as f64 + 1.0) * 0.25, (index as f64 - 2.0) * 0.125))
            .collect(),
    );
    let adjoint_input = DenseMatrix::from_row_major_vec(
        operator.num_rows(),
        2,
        (0..operator.num_rows() * 2)
            .map(|index| Complex64::new((index as f64 - 1.0) * 0.2, (index as f64 + 3.0) * -0.1))
            .collect(),
    );

    let mut forward = DenseMatrix::zeros(operator.num_rows(), 2);
    operator
        .try_apply_block(&x, &mut forward)
        .map_err(|error| {
            format!(
                "{name}: forward block x={}x{} y={}x{}: {error:?}",
                x.nrows(),
                x.ncols(),
                forward.nrows(),
                forward.ncols()
            )
        })?;
    let mut transpose = DenseMatrix::zeros(operator.num_cols(), 2);
    operator
        .try_apply_transpose_block(&adjoint_input, &mut transpose)
        .map_err(|error| format!("{name}: transpose block: {error:?}"))?;
    let mut hermitian = DenseMatrix::zeros(operator.num_cols(), 2);
    operator
        .try_apply_hermitian_block(&adjoint_input, &mut hermitian)
        .map_err(|error| format!("{name}: Hermitian block: {error:?}"))?;

    for column in 0..2 {
        let expected_forward = operator
            .try_apply(&x.column_to_vec(column))
            .map_err(|error| format!("{name}: forward column: {error:?}"))?;
        let expected_transpose = operator
            .try_apply_transpose(&adjoint_input.column_to_vec(column))
            .map_err(|error| format!("{name}: transpose column: {error:?}"))?;
        let expected_hermitian = operator
            .try_apply_hermitian(&adjoint_input.column_to_vec(column))
            .map_err(|error| format!("{name}: Hermitian column: {error:?}"))?;
        for row in 0..operator.num_rows() {
            let error = (forward[[row, column]] - expected_forward[row]).norm();
            if error > TOLERANCE {
                return Err(format!("{name}: forward block mismatch {error:e}"));
            }
        }
        for row in 0..operator.num_cols() {
            let transpose_error = (transpose[[row, column]] - expected_transpose[row]).norm();
            let hermitian_error = (hermitian[[row, column]] - expected_hermitian[row]).norm();
            if transpose_error > TOLERANCE || hermitian_error > TOLERANCE {
                return Err(format!(
                    "{name}: adjoint block mismatch transpose={transpose_error:e} Hermitian={hermitian_error:e}"
                ));
            }
        }
        metrics.block_checks += 3;
    }
    Ok(())
}

fn check_operator(
    name: &str,
    operator: &dyn LinearOperator<Complex64>,
    metrics: &mut Metrics,
) -> Result<(), String> {
    if operator.transpose_capability() != AdjointCapability::Exact {
        return Err(format!("{name}: transpose capability is not exact"));
    }
    if operator.hermitian_capability() != AdjointCapability::Exact {
        return Err(format!("{name}: Hermitian capability is not exact"));
    }

    for seed in 0..32_u64 {
        let x = property_vector(operator.num_cols(), 0x000A_11CE_0000_u64 + seed);
        let y = property_vector(operator.num_rows(), 0x000B_00B5_0000_u64 + seed);
        let forward = operator
            .try_apply(&x)
            .map_err(|error| format!("{name}: forward seed {seed}: {error:?}"))?;
        let transpose = operator
            .try_apply_transpose(&y)
            .map_err(|error| format!("{name}: transpose seed {seed}: {error:?}"))?;
        let hermitian = operator
            .try_apply_hermitian(&y)
            .map_err(|error| format!("{name}: Hermitian seed {seed}: {error:?}"))?;
        let forward_inner: Complex64 = forward.iter().zip(&y).map(|(a, b)| a.conj() * *b).sum();
        let hermitian_inner: Complex64 = x.iter().zip(&hermitian).map(|(a, b)| a.conj() * *b).sum();
        let transpose_left: Complex64 = forward.iter().zip(&y).map(|(a, b)| *a * *b).sum();
        let transpose_right: Complex64 = x.iter().zip(&transpose).map(|(a, b)| *a * *b).sum();
        let hermitian_error = (forward_inner - hermitian_inner).norm();
        let transpose_error = (transpose_left - transpose_right).norm();
        metrics.max_hermitian_error = metrics.max_hermitian_error.max(hermitian_error);
        metrics.max_transpose_error = metrics.max_transpose_error.max(transpose_error);
        let hermitian_scale = 1.0 + forward_inner.norm().max(hermitian_inner.norm());
        let transpose_scale = 1.0 + transpose_left.norm().max(transpose_right.norm());
        if hermitian_error > TOLERANCE * hermitian_scale {
            return Err(format!(
                "{name}: Hermitian identity failed at seed {seed}: {hermitian_error:e}"
            ));
        }
        if transpose_error > TOLERANCE * transpose_scale {
            return Err(format!(
                "{name}: transpose identity failed at seed {seed}: {transpose_error:e}"
            ));
        }
        metrics.identity_checks += 2;
    }
    check_block_parity(name, operator, metrics)
}

fn expect_unsupported<T>(label: &str, result: Result<T, SolverError>) -> Result<(), String> {
    match result {
        Err(SolverError::UnsupportedOperatorOperation { .. }) => Ok(()),
        Err(error) => Err(format!(
            "{label}: expected unsupported-operation error, got {error:?}"
        )),
        Ok(_) => Err(format!(
            "{label}: unsupported operation unexpectedly succeeded"
        )),
    }
}

fn check_rejection_contracts(metrics: &mut Metrics) -> Result<(), String> {
    let forward_only = FnOperator::<Complex64>::new_into(1, 1, |_x, y| {
        y[0] = Complex64::new(1.0, 0.0);
    });
    expect_unsupported(
        "forward-only transpose",
        forward_only.try_apply_transpose(&[Complex64::new(1.0, 0.0)]),
    )?;
    expect_unsupported(
        "forward-only Hermitian",
        forward_only.try_apply_hermitian(&[Complex64::new(1.0, 0.0)]),
    )?;
    metrics.rejection_checks += 2;

    let malformed = DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
        2,
        3,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    ));
    let bad_input = DenseMatrix::zeros(2, 1);
    let mut output = DenseMatrix::zeros(2, 1);
    match malformed.try_apply_block(&bad_input, &mut output) {
        Err(SolverError::DimensionMismatch { .. }) => {}
        Err(error) => return Err(format!("malformed block: wrong error {error:?}")),
        Ok(()) => return Err("malformed block unexpectedly succeeded".into()),
    }
    metrics.rejection_checks += 1;

    let non_finite = FnOperator::new_into_with_adjoint(
        1,
        1,
        |_x, y| y[0] = Complex64::new(1.0, 0.0),
        |_x, y| y[0] = Complex64::new(f64::NAN, 0.0),
        |_x, y| y[0] = Complex64::new(1.0, 0.0),
    );
    match non_finite.try_apply_transpose(&[Complex64::new(1.0, 0.0)]) {
        Err(SolverError::OperationFailed { .. }) => {}
        Err(error) => return Err(format!("non-finite callback: wrong error {error:?}")),
        Ok(_) => return Err("non-finite callback unexpectedly succeeded".into()),
    }
    metrics.rejection_checks += 1;
    Ok(())
}

fn run() -> Result<Metrics, String> {
    let csr = fixture();
    let compact = CompactCsrMatrix::from_csr(&csr)
        .ok_or_else(|| "compact CSR: fixture exceeds compact index range".to_string())?;
    let bucketed = RowBucketedCsrMatrix::from_csr(&csr);
    let sell = SellCSigmaMatrix::from_csr(&csr, 2, 8);

    let a = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.25, -1.0),
            Complex64::new(3.0, -0.75),
        ],
    );
    let b = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex64::new(-1.0, 0.25),
            Complex64::new(0.5, 1.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(1.0, 1.0),
        ],
    );
    let sum = sonium_solvers_sparse::SumOperator::new(&a, &b);
    let scaled = sonium_solvers_sparse::ScaledOperator::new(&sum, Complex64::new(0.5, -1.0));
    let composite = sonium_solvers_sparse::CompositeOperator::new(&scaled, &a);
    let matrix_free = FnOperator::new_into_with_adjoint(
        2,
        2,
        |x, y| {
            y[0] = x[0] + Complex64::new(2.0, 0.0) * x[1];
            y[1] = Complex64::new(-1.0, 0.5) * x[0] + x[1];
        },
        |x, y| {
            y[0] = x[0] + Complex64::new(-1.0, 0.5) * x[1];
            y[1] = Complex64::new(2.0, 0.0) * x[0] + x[1];
        },
        |x, y| {
            y[0] = x[0] + Complex64::new(-1.0, -0.5) * x[1];
            y[1] = Complex64::new(2.0, 0.0) * x[0] + x[1];
        },
    );
    let dense = DenseBlockOperator::new(DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 1.0),
            Complex64::new(2.0, -0.75),
            Complex64::new(0.25, 0.5),
        ],
    ));

    let operators: [(&str, &dyn LinearOperator<Complex64>); 9] = [
        ("CSR", &csr),
        ("compact CSR", &compact),
        ("row-bucketed CSR", &bucketed),
        ("SELL-C-sigma", &sell),
        ("sum", &sum),
        ("scaled", &scaled),
        ("composite", &composite),
        ("matrix-free", &matrix_free),
        ("dense block", &dense),
    ];
    let mut metrics = Metrics::default();
    for (name, operator) in operators {
        check_operator(name, operator, &mut metrics)?;
    }
    check_rejection_contracts(&mut metrics)?;
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:operator-adjoint-randomized-v1")
                .reference("independent transpose/Hermitian inner-product and block parity recomputation")
                .reference_checksum("analytic/deterministic-fixture:operator-adjoint-randomized-v1")
                .dimension("CSR, Compact CSR, Row-Bucketed CSR, SELL-C-sigma, composed, matrix-free, dense Complex64 operators")
                .tolerance("maximum scaled transpose/Hermitian identity error <= 1e-10")
                .metric("max_hermitian_error", metrics.max_hermitian_error)
                .metric("max_transpose_error", metrics.max_transpose_error)
                .metric("identity_checks", metrics.identity_checks as f64)
                .metric("block_checks", metrics.block_checks as f64)
                .metric("rejection_checks", metrics.rejection_checks as f64)
                .maximum("max_hermitian_error", TOLERANCE)
                .maximum("max_transpose_error", TOLERANCE)
                .seed("solvers-operator-adjoint-randomized:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum("analytic/deterministic-fixture:operator-adjoint-randomized-v1")
                .reference_checksum("analytic/deterministic-fixture:operator-adjoint-randomized-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
