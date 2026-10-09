//! Independent CPU reference for the public compatibility facade's CG path.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use sonium_solvers::{
    ComplexField, CsrBuilder, LinearSolveConfig, LinearSolverType, SolverStatus, solve,
};

const N: usize = 4;

const REAL: [[f64; N]; N] = [
    [6.0, 1.0, 0.0, 0.0],
    [1.0, 5.0, 1.0, 0.0],
    [0.0, 1.0, 4.0, 1.0],
    [0.0, 0.0, 1.0, 3.0],
];

// The imaginary part is skew-symmetric, so the complex fixture is Hermitian.
const IMAG: [[f64; N]; N] = [
    [0.0, 0.20, 0.0, 0.0],
    [-0.20, 0.0, -0.15, 0.0],
    [0.0, 0.15, 0.0, 0.10],
    [0.0, 0.0, -0.10, 0.0],
];

const X_REAL: [f64; N] = [1.0, -0.5, 0.75, -1.25];
const X_IMAG: [f64; N] = [0.10, -0.20, 0.30, 0.15];

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

/// A small partial-pivot dense solve keeps the oracle independent of the
/// production CSR/Krylov implementation under test.
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
                    .expect("finite reference pivot")
            })
            .expect("non-empty reference pivot range");
        assert!(
            augmented[pivot][column].norm()
                > T::Real::from_f64(1.0e-12).expect("finite pivot threshold")
        );
        augmented.swap(column, pivot);

        let pivot_value = augmented[column][column];
        let pivot_row = augmented[column].clone();
        for row in (column + 1)..N {
            let factor = augmented[row][column] / pivot_value;
            augmented[row][column] = T::zero();
            for index in (column + 1)..=N {
                augmented[row][index] -= factor * pivot_row[index];
            }
        }
    }

    let mut solution = [T::zero(); N];
    for row in (0..N).rev() {
        let mut value = augmented[row][N];
        for column in (row + 1)..N {
            value -= augmented[row][column] * solution[column];
        }
        solution[row] = value / augmented[row][row];
    }
    solution
}

fn max_error<T: ComplexField>(actual: &[T], expected: &[T; N]) -> f64
where
    T::Real: ToPrimitive,
{
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected)
                .norm()
                .to_f64()
                .expect("finite solution error")
        })
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

fn run_scalar<T: ComplexField>(complex_content: bool, tolerance: f64) -> Result<f64, String>
where
    T::Real: FromPrimitive + ToPrimitive + Send + Sync,
    T: Send + Sync,
{
    let dense = matrix::<T>(complex_content);
    let expected = known_solution::<T>(complex_content);
    let rhs = dense_matvec(&dense, &expected);
    let reference = dense_reference(&dense, &rhs);
    let reference_error = max_error(&reference, &expected);
    if reference_error > tolerance {
        return Err(format!("dense reference error {reference_error:.3e}"));
    }

    let matrix = csr(&dense);
    let config = LinearSolveConfig {
        solver: LinearSolverType::Cg,
        max_iterations: 32,
        tolerance,
        restart: 8,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };
    let result = solve(&matrix, &rhs, &config).map_err(|error| error.to_string())?;
    if result.solver_status() != SolverStatus::Converged {
        return Err(format!("CG returned {:?}", result.solver_status()));
    }
    if result.accuracy_status() != sonium_solvers::AccuracyStatus::Attained {
        return Err(format!(
            "CG returned accuracy status {:?}",
            result.accuracy_status()
        ));
    }
    result
        .validate_contract()
        .map_err(|error| error.to_string())?;

    let solution_error = max_error(&result.x, &reference);
    let residual = relative_residual(&matrix, &result.x, &rhs);
    if solution_error > tolerance * 10.0 || residual > tolerance * 10.0 {
        return Err(format!(
            "CG error {solution_error:.3e}, residual {residual:.3e}, tolerance {tolerance:.3e}"
        ));
    }
    Ok(solution_error.max(residual))
}

pub fn run_case() -> Result<Vec<(&'static str, f64)>, String> {
    Ok(vec![
        ("f32_error", run_scalar::<f32>(false, 1.0e-4)?),
        ("f64_error", run_scalar::<f64>(false, 1.0e-12)?),
        ("complex32_error", run_scalar::<Complex32>(true, 1.0e-4)?),
        ("complex64_error", run_scalar::<Complex64>(true, 1.0e-12)?),
    ])
}

#[test]
fn high_level_cg_executes_all_supported_scalar_families() {
    run_case().expect("public CG compatibility slice should pass");
}
