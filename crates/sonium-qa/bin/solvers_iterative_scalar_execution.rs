//! CPU execution coverage for iterative solvers across supported scalar families.

use num_traits::{FromPrimitive, ToPrimitive};
use sonium_qa::QaResult;
use sonium_solvers::{
    BiCgstabConfig, CgConfig, CgsConfig, ComplexField, CsrBuilder, FgmresConfig, GmresConfig,
    IdrsConfig, IdrsStabConfig, MinresConfig, TfqmrConfig, try_bicgstab, try_cg, try_cgs,
    try_fgmres, try_gmres, try_idrs, try_idrs_stab, try_minres, try_tfqmr,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-scalar-execution.v1";
const N: usize = 6;
const MAX_ITERATIONS: usize = 100;

const REAL_PART: [[f64; N]; N] = [
    [5.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [2.0, 6.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 5.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 7.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 4.0],
];
const SPD_REAL_PART: [[f64; N]; N] = [
    [4.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 4.0, -1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 4.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0, 4.0, -1.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, 4.0],
];
const IMAG_PART: [[f64; N]; N] = [
    [0.20, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -0.15, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.10, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
];
const SOLUTION_REAL: [f64; N] = [1.0, -0.5, 0.75, 1.5, -1.25, 0.25];
const SOLUTION_IMAG: [f64; N] = [0.10, -0.20, 0.30, -0.40, 0.50, -0.60];

fn scalar<T: ComplexField>(real: f64, imag: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_re_im(
        <T::Real as FromPrimitive>::from_f64(real).expect("finite real fixture"),
        <T::Real as FromPrimitive>::from_f64(imag).expect("finite imaginary fixture"),
    )
}

fn dense_matrix<T: ComplexField>(complex_content: bool) -> [[T; N]; N]
where
    T::Real: FromPrimitive,
{
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            scalar(
                REAL_PART[row][column],
                if complex_content {
                    IMAG_PART[row][column]
                } else {
                    0.0
                },
            )
        })
    })
}
fn spd_matrix<T: ComplexField>(_complex_content: bool) -> [[T; N]; N]
where
    T::Real: FromPrimitive,
{
    std::array::from_fn(|row| std::array::from_fn(|column| scalar(SPD_REAL_PART[row][column], 0.0)))
}

fn known_solution<T: ComplexField>(complex_content: bool) -> [T; N]
where
    T::Real: FromPrimitive,
{
    std::array::from_fn(|index| {
        scalar(
            SOLUTION_REAL[index],
            if complex_content {
                SOLUTION_IMAG[index]
            } else {
                0.0
            },
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

fn csr_matrix<T: ComplexField>(matrix: &[[T; N]; N]) -> sonium_solvers::CsrMatrix<T> {
    let mut builder = CsrBuilder::new(N, N);
    for values in matrix {
        builder.add_row_entries(values.iter().enumerate().filter_map(|(column, value)| {
            (value.norm() > scalar::<T>(0.0, 0.0).norm()).then_some((column, *value))
        }));
    }
    builder.finish()
}

/// Small independent reference solve with partial pivoting.
fn dense_reference<T: ComplexField>(matrix: &[[T; N]; N], rhs: &[T; N]) -> [T; N]
where
    T::Real: FromPrimitive,
{
    let mut augmented: [[T; N + 1]; N] = std::array::from_fn(|row| {
        let mut values = [T::zero(); N + 1];
        values[..N].copy_from_slice(&matrix[row]);
        values[N] = rhs[row];
        values
    });

    for column in 0..N {
        let pivot = (column..N)
            .max_by(|&left, &right| {
                augmented[left][column]
                    .norm()
                    .partial_cmp(&augmented[right][column].norm())
                    .expect("finite pivot")
            })
            .expect("non-empty pivot range");
        if augmented[pivot][column].norm() <= scalar::<T>(1.0e-7, 0.0).norm() {
            panic!("singular scalar execution fixture");
        }
        augmented.swap(column, pivot);

        let diagonal = augmented[column][column];
        for value in &mut augmented[column][column..=N] {
            *value /= diagonal;
        }
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            for index in column..=N {
                augmented[row][index] -= factor * augmented[column][index];
            }
        }
    }

    std::array::from_fn(|row| augmented[row][N])
}

fn max_error<T: ComplexField>(left: &[T], right: &[T]) -> f64
where
    T::Real: ToPrimitive,
{
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left - *right).norm().to_f64().expect("finite error"))
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
    residual / rhs_norm
}

fn check_result<T: ComplexField>(
    method: &'static str,
    label: &'static str,
    metric_names: (&'static str, &'static str, &'static str),
    solution: &[T],
    converged: bool,
    iterations: usize,
    reference: &[T],
    matrix: &sonium_solvers::CsrMatrix<T>,
    rhs: &[T],
    tolerance: f64,
) -> Result<Vec<(&'static str, f64)>, String>
where
    T::Real: ToPrimitive,
{
    if !converged {
        return Err(format!(
            "{label} {method} did not converge after {iterations} iterations"
        ));
    }
    let error = max_error(solution, reference);
    let residual = relative_residual(matrix, solution, rhs);
    if !error.is_finite() || !residual.is_finite() || error > tolerance || residual > tolerance {
        return Err(format!(
            "{label} {method} failed scalar execution gate: error={error:.3e}, residual={residual:.3e}, tolerance={tolerance:.3e}"
        ));
    }
    Ok(vec![
        (metric_names.0, error),
        (metric_names.1, residual),
        (metric_names.2, iterations as f64),
    ])
}

fn run_scalar<T: ComplexField>(
    label: &'static str,
    tolerance: f64,
    complex_content: bool,
) -> Result<Vec<(&'static str, f64)>, String>
where
    T::Real: FromPrimitive + ToPrimitive,
{
    let dense = dense_matrix::<T>(complex_content);
    let known = known_solution::<T>(complex_content);
    let rhs = dense_matvec(&dense, &known);
    let reference = dense_reference(&dense, &rhs);
    let matrix = csr_matrix(&dense);
    let reference_error = max_error(&reference, &known);
    if reference_error > tolerance {
        return Err(format!(
            "{label} dense reference mismatch: {reference_error:.3e}"
        ));
    }

    let real_tolerance =
        <T::Real as FromPrimitive>::from_f64(tolerance).expect("finite scalar tolerance");
    // Tighten each algorithm's stopping estimate so the independent
    // reference/residual gate is not masked by method-specific estimates.
    let solver_tolerance =
        real_tolerance / <T::Real as FromPrimitive>::from_f64(10.0).expect("finite scale");
    let gmres_config = GmresConfig {
        max_iterations: MAX_ITERATIONS,
        restart: N,
        tolerance: solver_tolerance,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let fgmres_config = FgmresConfig {
        max_iterations: MAX_ITERATIONS,
        restart: N,
        tolerance: solver_tolerance,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
    };
    let cgs_config = CgsConfig {
        max_iterations: MAX_ITERATIONS,
        tolerance: solver_tolerance,
        print_interval: 0,
    };
    let idrs_config = IdrsConfig {
        s: 2,
        max_iterations: MAX_ITERATIONS,
        tolerance: solver_tolerance,
        print_interval: 0,
    };
    let idrs_stab_config = IdrsStabConfig {
        s: 2,
        ell: 2,
        max_iterations: MAX_ITERATIONS,
        tolerance: solver_tolerance,
        print_interval: 0,
    };
    let tfqmr_config = TfqmrConfig {
        max_iterations: MAX_ITERATIONS,
        tolerance: solver_tolerance,
        print_interval: 0,
    };
    let bicgstab_config = BiCgstabConfig {
        max_iterations: MAX_ITERATIONS,
        tolerance: solver_tolerance,
        print_interval: 0,
    };

    let gmres = try_gmres(&matrix, &rhs, &gmres_config)
        .map_err(|error| format!("{label} GMRES returned checked error: {error:?}"))?;
    let fgmres = try_fgmres(&matrix, &rhs, &fgmres_config)
        .map_err(|error| format!("{label} FGMRES returned checked error: {error:?}"))?;
    let bicgstab = try_bicgstab(&matrix, &rhs, &bicgstab_config)
        .map_err(|error| format!("{label} BiCGSTAB returned checked error: {error:?}"))?;
    let cgs = try_cgs(&matrix, &rhs, &cgs_config)
        .map_err(|error| format!("{label} CGS returned checked error: {error:?}"))?;
    let idrs = try_idrs(&matrix, &rhs, &idrs_config)
        .map_err(|error| format!("{label} IDR(s) returned checked error: {error:?}"))?;
    let idrs_stab = try_idrs_stab(&matrix, &rhs, &idrs_stab_config)
        .map_err(|error| format!("{label} IDR(s)stab returned checked error: {error:?}"))?;
    let tfqmr = try_tfqmr(&matrix, &rhs, &tfqmr_config)
        .map_err(|error| format!("{label} TFQMR returned checked error: {error:?}"))?;

    let (
        reference_metric,
        gmres_metrics,
        fgmres_metrics,
        bicgstab_metrics,
        cgs_metrics,
        idrs_metrics,
        idrs_stab_metrics,
        tfqmr_metrics,
    ) = match label {
        "f32" => (
            "f32_dense_reference_error",
            (
                "f32_gmres_reference_error",
                "f32_gmres_relative_residual",
                "f32_gmres_iterations",
            ),
            (
                "f32_fgmres_reference_error",
                "f32_fgmres_relative_residual",
                "f32_fgmres_iterations",
            ),
            (
                "f32_bicgstab_reference_error",
                "f32_bicgstab_relative_residual",
                "f32_bicgstab_iterations",
            ),
            (
                "f32_cgs_reference_error",
                "f32_cgs_relative_residual",
                "f32_cgs_iterations",
            ),
            (
                "f32_idrs_reference_error",
                "f32_idrs_relative_residual",
                "f32_idrs_iterations",
            ),
            (
                "f32_idrs_stab_reference_error",
                "f32_idrs_stab_relative_residual",
                "f32_idrs_stab_iterations",
            ),
            (
                "f32_tfqmr_reference_error",
                "f32_tfqmr_relative_residual",
                "f32_tfqmr_iterations",
            ),
        ),
        "f64" => (
            "f64_dense_reference_error",
            (
                "f64_gmres_reference_error",
                "f64_gmres_relative_residual",
                "f64_gmres_iterations",
            ),
            (
                "f64_fgmres_reference_error",
                "f64_fgmres_relative_residual",
                "f64_fgmres_iterations",
            ),
            (
                "f64_bicgstab_reference_error",
                "f64_bicgstab_relative_residual",
                "f64_bicgstab_iterations",
            ),
            (
                "f64_cgs_reference_error",
                "f64_cgs_relative_residual",
                "f64_cgs_iterations",
            ),
            (
                "f64_idrs_reference_error",
                "f64_idrs_relative_residual",
                "f64_idrs_iterations",
            ),
            (
                "f64_idrs_stab_reference_error",
                "f64_idrs_stab_relative_residual",
                "f64_idrs_stab_iterations",
            ),
            (
                "f64_tfqmr_reference_error",
                "f64_tfqmr_relative_residual",
                "f64_tfqmr_iterations",
            ),
        ),
        "complex32" => (
            "complex32_dense_reference_error",
            (
                "complex32_gmres_reference_error",
                "complex32_gmres_relative_residual",
                "complex32_gmres_iterations",
            ),
            (
                "complex32_fgmres_reference_error",
                "complex32_fgmres_relative_residual",
                "complex32_fgmres_iterations",
            ),
            (
                "complex32_bicgstab_reference_error",
                "complex32_bicgstab_relative_residual",
                "complex32_bicgstab_iterations",
            ),
            (
                "complex32_cgs_reference_error",
                "complex32_cgs_relative_residual",
                "complex32_cgs_iterations",
            ),
            (
                "complex32_idrs_reference_error",
                "complex32_idrs_relative_residual",
                "complex32_idrs_iterations",
            ),
            (
                "complex32_idrs_stab_reference_error",
                "complex32_idrs_stab_relative_residual",
                "complex32_idrs_stab_iterations",
            ),
            (
                "complex32_tfqmr_reference_error",
                "complex32_tfqmr_relative_residual",
                "complex32_tfqmr_iterations",
            ),
        ),
        "complex64" => (
            "complex64_dense_reference_error",
            (
                "complex64_gmres_reference_error",
                "complex64_gmres_relative_residual",
                "complex64_gmres_iterations",
            ),
            (
                "complex64_fgmres_reference_error",
                "complex64_fgmres_relative_residual",
                "complex64_fgmres_iterations",
            ),
            (
                "complex64_bicgstab_reference_error",
                "complex64_bicgstab_relative_residual",
                "complex64_bicgstab_iterations",
            ),
            (
                "complex64_cgs_reference_error",
                "complex64_cgs_relative_residual",
                "complex64_cgs_iterations",
            ),
            (
                "complex64_idrs_reference_error",
                "complex64_idrs_relative_residual",
                "complex64_idrs_iterations",
            ),
            (
                "complex64_idrs_stab_reference_error",
                "complex64_idrs_stab_relative_residual",
                "complex64_idrs_stab_iterations",
            ),
            (
                "complex64_tfqmr_reference_error",
                "complex64_tfqmr_relative_residual",
                "complex64_tfqmr_iterations",
            ),
        ),
        _ => unreachable!("all scalar labels are listed"),
    };
    let mut metrics = vec![(reference_metric, reference_error)];
    metrics.extend(check_result(
        "gmres",
        label,
        gmres_metrics,
        &gmres.x,
        gmres.converged,
        gmres.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "fgmres",
        label,
        fgmres_metrics,
        &fgmres.x,
        fgmres.converged,
        fgmres.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "bicgstab",
        label,
        bicgstab_metrics,
        &bicgstab.x,
        bicgstab.converged,
        bicgstab.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "cgs",
        label,
        cgs_metrics,
        &cgs.x,
        cgs.converged,
        cgs.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "idrs",
        label,
        idrs_metrics,
        &idrs.x,
        idrs.converged,
        idrs.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "idrs_stab",
        label,
        idrs_stab_metrics,
        &idrs_stab.x,
        idrs_stab.converged,
        idrs_stab.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "tfqmr",
        label,
        tfqmr_metrics,
        &tfqmr.x,
        tfqmr.converged,
        tfqmr.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    Ok(metrics)
}

fn run_spd_scalar<T: ComplexField>(
    label: &'static str,
    tolerance: f64,
    complex_content: bool,
) -> Result<Vec<(&'static str, f64)>, String>
where
    T::Real: FromPrimitive + ToPrimitive,
{
    let dense = spd_matrix::<T>(complex_content);
    let known = known_solution::<T>(complex_content);
    let rhs = dense_matvec(&dense, &known);
    let reference = dense_reference(&dense, &rhs);
    let matrix = csr_matrix(&dense);
    let reference_error = max_error(&reference, &known);
    if reference_error > tolerance {
        return Err(format!(
            "{label} SPD dense reference mismatch: {reference_error:.3e}"
        ));
    }
    let real_tolerance =
        <T::Real as FromPrimitive>::from_f64(tolerance).expect("finite scalar tolerance");
    let solver_tolerance =
        real_tolerance / <T::Real as FromPrimitive>::from_f64(10.0).expect("finite scale");
    let cg = try_cg(
        &matrix,
        &rhs,
        &CgConfig {
            max_iterations: MAX_ITERATIONS,
            tolerance: solver_tolerance,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("{label} CG returned checked error: {error:?}"))?;
    let minres = try_minres(
        &matrix,
        &rhs,
        &MinresConfig {
            max_iterations: MAX_ITERATIONS,
            tolerance: solver_tolerance,
            print_interval: 0,
            condition_limit: <T::Real as FromPrimitive>::from_f64(1.0e14)
                .expect("finite condition limit"),
        },
    )
    .map_err(|error| format!("{label} MINRES returned checked error: {error:?}"))?;
    let (reference_metric, cg_metrics, minres_metrics) = match label {
        "f32" => (
            "f32_spd_dense_reference_error",
            (
                "f32_cg_reference_error",
                "f32_cg_relative_residual",
                "f32_cg_iterations",
            ),
            (
                "f32_minres_reference_error",
                "f32_minres_relative_residual",
                "f32_minres_iterations",
            ),
        ),
        "f64" => (
            "f64_spd_dense_reference_error",
            (
                "f64_cg_reference_error",
                "f64_cg_relative_residual",
                "f64_cg_iterations",
            ),
            (
                "f64_minres_reference_error",
                "f64_minres_relative_residual",
                "f64_minres_iterations",
            ),
        ),
        "complex32" => (
            "complex32_spd_dense_reference_error",
            (
                "complex32_cg_reference_error",
                "complex32_cg_relative_residual",
                "complex32_cg_iterations",
            ),
            (
                "complex32_minres_reference_error",
                "complex32_minres_relative_residual",
                "complex32_minres_iterations",
            ),
        ),
        "complex64" => (
            "complex64_spd_dense_reference_error",
            (
                "complex64_cg_reference_error",
                "complex64_cg_relative_residual",
                "complex64_cg_iterations",
            ),
            (
                "complex64_minres_reference_error",
                "complex64_minres_relative_residual",
                "complex64_minres_iterations",
            ),
        ),
        _ => unreachable!("all scalar labels are listed"),
    };
    let mut metrics = vec![(reference_metric, reference_error)];
    metrics.extend(check_result(
        "cg",
        label,
        cg_metrics,
        &cg.x,
        cg.converged,
        cg.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    metrics.extend(check_result(
        "minres",
        label,
        minres_metrics,
        &minres.x,
        minres.converged,
        minres.iterations,
        &reference,
        &matrix,
        &rhs,
        tolerance,
    )?);
    Ok(metrics)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let mut metrics = Vec::new();
    metrics.extend(run_scalar::<f32>("f32", 1.0e-4, false)?);
    metrics.extend(run_scalar::<f64>("f64", 1.0e-12, false)?);
    metrics.extend(run_scalar::<num_complex::Complex32>(
        "complex32",
        1.0e-4,
        true,
    )?);
    metrics.extend(run_scalar::<num_complex::Complex64>(
        "complex64",
        1.0e-12,
        true,
    )?);
    metrics.extend(run_spd_scalar::<f32>("f32", 1.0e-4, false)?);
    metrics.extend(run_spd_scalar::<f64>("f64", 1.0e-12, false)?);
    metrics.extend(run_spd_scalar::<num_complex::Complex32>(
        "complex32",
        1.0e-4,
        true,
    )?);
    metrics.extend(run_spd_scalar::<num_complex::Complex64>(
        "complex64",
        1.0e-12,
        true,
    )?);
    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .reference(
                    "independent partial-pivot dense solve from deterministic known-solution fixtures",
                )
                .reference_checksum("analytic/deterministic-fixture:iterative-scalar-execution-v1")
                .input_checksum("analytic/deterministic-fixture:iterative-scalar-execution-v1")
                .dimension("6x6 real nonsymmetric and complex nonsymmetric CSR systems")
                .tolerance(
                    "all scalar-family reference errors and independently recomputed relative residuals remain within f32/Complex32 1e-4 or f64/Complex64 1e-12",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            let result = metrics
                .into_iter()
                .fold(result, |result, (name, value)| result.metric(name, value));
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
