//! CPU execution coverage for the high-level generic solver facade.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use sonium_qa::QaResult;
use sonium_solvers::{ComplexField, CsrBuilder, LinearSolveConfig, LinearSolverType, solve};
use std::time::Instant;

const CASE_ID: &str = "solvers.compatibility-scalar-execution.v1";
const N: usize = 5;

const REAL: [[f64; N]; N] = [
    [6.0, 1.0, 0.0, 0.0, 0.0],
    [2.0, 7.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, 5.0, 1.0, 0.0],
    [0.0, 0.0, 2.0, 6.0, 1.0],
    [1.0, 0.0, 0.0, -1.0, 4.0],
];
const IMAG: [[f64; N]; N] = [
    [0.15, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -0.10, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.12, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.08],
];
const X_REAL: [f64; N] = [1.0, -0.5, 0.75, 1.25, -0.8];
const X_IMAG: [f64; N] = [0.1, -0.2, 0.3, -0.4, 0.5];

fn scalar<T: ComplexField>(real: f64, imag: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_re_im(
        T::Real::from_f64(real).expect("finite real fixture"),
        T::Real::from_f64(imag).expect("finite imaginary fixture"),
    )
}

fn matrix<T: ComplexField>(complex_content: bool) -> [[T; N]; N]
where
    T::Real: FromPrimitive,
{
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            scalar(
                REAL[row][column],
                if complex_content {
                    IMAG[row][column]
                } else {
                    0.0
                },
            )
        })
    })
}

fn known_solution<T: ComplexField>(complex_content: bool) -> [T; N]
where
    T::Real: FromPrimitive,
{
    std::array::from_fn(|index| {
        scalar(
            X_REAL[index],
            if complex_content { X_IMAG[index] } else { 0.0 },
        )
    })
}

fn dense_matvec<T: ComplexField>(matrix: &[[T; N]; N], x: &[T; N]) -> [T; N] {
    std::array::from_fn(|row| {
        matrix[row]
            .iter()
            .zip(x)
            .fold(T::zero(), |sum, (a, value)| sum + *a * *value)
    })
}

fn csr<T: ComplexField>(matrix: &[[T; N]; N]) -> sonium_solvers::CsrMatrix<T> {
    let mut builder = CsrBuilder::new(N, N);
    for row in matrix {
        builder.add_row_entries(
            row.iter()
                .enumerate()
                .filter(|(_, value)| value.norm() > T::Real::zero())
                .map(|(column, &value)| (column, value)),
        );
    }
    builder.finish()
}

fn dense_reference<T: ComplexField>(matrix: &[[T; N]; N], rhs: &[T; N]) -> [T; N]
where
    T::Real: FromPrimitive,
{
    let mut augmented = vec![vec![T::zero(); N + 1]; N];
    for row in 0..N {
        augmented[row][..N].copy_from_slice(&matrix[row]);
        augmented[row][N] = rhs[row];
    }
    for column in 0..N {
        let pivot = (column..N)
            .max_by(|&left, &right| {
                augmented[left][column]
                    .norm()
                    .partial_cmp(&augmented[right][column].norm())
                    .expect("finite pivot")
            })
            .expect("non-empty pivot range");
        assert!(augmented[pivot][column].norm() > scalar::<T>(1.0e-8, 0.0).norm());
        augmented.swap(column, pivot);
        let diagonal = augmented[column][column];
        for value in &mut augmented[column][column..=N] {
            *value /= diagonal;
        }
        let pivot_row = augmented[column].clone();
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            for index in column..=N {
                augmented[row][index] -= factor * pivot_row[index];
            }
        }
    }
    std::array::from_fn(|row| augmented[row][N])
}

fn max_error<T: ComplexField>(actual: &[T], expected: &[T; N]) -> f64
where
    T::Real: ToPrimitive,
{
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm().to_f64().expect("finite error"))
        .fold(0.0, f64::max)
}

fn relative_residual<T: ComplexField>(
    matrix: &sonium_solvers::CsrMatrix<T>,
    solution: &[T],
    rhs: &[T],
) -> f64
where
    T::Real: ToPrimitive,
{
    let residual = matrix
        .matvec(solution)
        .into_iter()
        .zip(rhs)
        .map(|(actual, expected)| {
            (actual - *expected)
                .norm()
                .to_f64()
                .expect("finite residual")
        })
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs
        .iter()
        .map(|value| value.norm().to_f64().expect("finite rhs"))
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    residual / rhs_norm.max(1.0e-30)
}

fn run_scalar<T: ComplexField>(
    label: &'static str,
    complex_content: bool,
    tolerance: f64,
) -> Result<Vec<(&'static str, f64)>, String>
where
    T::Real: FromPrimitive + ToPrimitive,
    T: Send + Sync,
{
    let dense = matrix::<T>(complex_content);
    let known = known_solution::<T>(complex_content);
    let rhs = dense_matvec(&dense, &known);
    let reference = dense_reference(&dense, &rhs);
    let matrix = csr(&dense);
    let reference_error = max_error(&reference, &known);
    if reference_error > tolerance {
        return Err(format!("{label} reference error {reference_error:.3e}"));
    }

    let result = solve(
        &matrix,
        &rhs,
        &LinearSolveConfig {
            solver: LinearSolverType::Gmres,
            max_iterations: 200,
            tolerance,
            restart: N,
            exact_residual_check_interval: 1,
            verbosity: 0,
        },
    )
    .map_err(|error| format!("{label} high-level solve failed: {error:?}"))?;
    result
        .validate_contract()
        .map_err(|error| format!("{label} result contract failed: {error:?}"))?;
    if !result.converged {
        return Err(format!("{label} high-level solve did not converge"));
    }
    let error = max_error(&result.x, &reference);
    let residual = relative_residual(&matrix, &result.x, &rhs);
    if !error.is_finite() || !residual.is_finite() || error > tolerance || residual > tolerance {
        return Err(format!(
            "{label} facade gate failed: error={error:.3e}, residual={residual:.3e}, tolerance={tolerance:.3e}"
        ));
    }

    let names = match label {
        "f32" => (
            "f32_reference_error",
            "f32_relative_residual",
            "f32_iterations",
        ),
        "f64" => (
            "f64_reference_error",
            "f64_relative_residual",
            "f64_iterations",
        ),
        "complex32" => (
            "complex32_reference_error",
            "complex32_relative_residual",
            "complex32_iterations",
        ),
        "complex64" => (
            "complex64_reference_error",
            "complex64_relative_residual",
            "complex64_iterations",
        ),
        _ => unreachable!("all scalar labels are listed"),
    };
    Ok(vec![
        (names.0, error),
        (names.1, residual),
        (names.2, result.iterations as f64),
    ])
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let mut metrics = Vec::new();
    metrics.extend(run_scalar::<f32>("f32", false, 1.0e-4)?);
    metrics.extend(run_scalar::<f64>("f64", false, 1.0e-12)?);
    metrics.extend(run_scalar::<Complex32>("complex32", true, 1.0e-4)?);
    metrics.extend(run_scalar::<Complex64>("complex64", true, 1.0e-12)?);
    Ok(metrics)
}

fn metric_limit(name: &str) -> f64 {
    if name.ends_with("iterations") {
        200.0
    } else if name.starts_with("f32_") || name.starts_with("complex32_") {
        1.0e-4
    } else {
        1.0e-12
    }
}

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(run);
    let (passed, result) = match outcome {
        Ok(Ok(metrics)) => {
            let result = metrics.into_iter().fold(
                QaResult::passed(CASE_ID)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .reference("independent partial-pivot dense references and CSR residuals")
                    .reference_checksum(
                        "analytic/deterministic-fixture:compatibility-scalar-execution-v1",
                    )
                    .input_checksum(
                        "analytic/deterministic-fixture:compatibility-scalar-execution-v1",
                    )
                    .dimension("5x5 real/complex nonsymmetric CSR through generic solve<T>")
                    .formulation("public high-level checked compatibility facade")
                    .tolerance("f32/Complex32 <= 1e-4; f64/Complex64 <= 1e-12")
                    .elapsed_ms(started.elapsed().as_millis() as u64),
                |result, (name, value)| {
                    result.metric(name, value).maximum(name, metric_limit(name))
                },
            );
            (true, result)
        }
        Ok(Err(error)) => (
            false,
            QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64),
        ),
        Err(payload) => (
            false,
            QaResult::failed(
                CASE_ID,
                format!("compatibility scalar packet panicked: {payload:?}"),
            )
            .elapsed_ms(started.elapsed().as_millis() as u64),
        ),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
