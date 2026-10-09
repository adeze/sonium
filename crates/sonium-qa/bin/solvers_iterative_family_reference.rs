//! Independent CPU reference for the checked iterative solver family.

use sonium_qa::QaResult;
use sonium_solvers::{
    BiCgstabConfig, CgsConfig, CsrBuilder, FgmresConfig, GmresConfig, IdrsConfig, IdrsStabConfig,
    MinresConfig, try_bicgstab, try_cgs, try_fgmres, try_gmres, try_idrs, try_idrs_stab,
    try_minres,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-family-reference.v1";
const N: usize = 6;
const TOLERANCE: f64 = 1.0e-10;

const MATRIX: [[f64; N]; N] = [
    [5.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [2.0, 6.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 5.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 7.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 4.0],
];

const KNOWN_SOLUTION: [f64; N] = [1.0, -0.5, 0.75, 1.5, -1.25, 0.25];
const MINRES_DIAGONAL: [f64; 8] = [-3.0, -1.0, 1.0, 3.0, -2.0, 2.0, -4.0, 4.0];
const MINRES_SOLUTION: [f64; 8] = [1.0, -0.5, 0.75, -1.25, 1.5, -1.75, 2.0, -2.25];

fn dense_matvec(x: &[f64]) -> Vec<f64> {
    MATRIX
        .iter()
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

fn csr_matrix() -> sonium_solvers::CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(N, N);
    for row in MATRIX {
        builder.add_row_entries(
            row.into_iter()
                .enumerate()
                .filter(|(_, value)| *value != 0.0)
                .map(|(column, value)| (column, value)),
        );
    }
    builder.finish()
}

fn diagonal_csr(diagonal: &[f64]) -> sonium_solvers::CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(diagonal.len(), diagonal.len());
    for (row, &value) in diagonal.iter().enumerate() {
        builder.add_row_entries([(row, value)].into_iter());
    }
    builder.finish()
}

/// Small independent dense reference with partial pivoting.
fn dense_reference(rhs: &[f64]) -> Result<Vec<f64>, String> {
    let mut augmented = vec![vec![0.0; N + 1]; N];
    for row in 0..N {
        augmented[row][..N].copy_from_slice(&MATRIX[row]);
        augmented[row][N] = rhs[row];
    }

    for column in 0..N {
        let pivot = (column..N)
            .max_by(|&left, &right| {
                augmented[left][column]
                    .abs()
                    .total_cmp(&augmented[right][column].abs())
            })
            .ok_or_else(|| "dense reference has no pivot".to_string())?;
        if augmented[pivot][column].abs() <= 1.0e-14 {
            return Err(format!("dense reference singular at column {column}"));
        }
        augmented.swap(column, pivot);
        let pivot_value = augmented[column][column];
        for entry in &mut augmented[column][column..=N] {
            *entry /= pivot_value;
        }
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor == 0.0 {
                continue;
            }
            for col in column..=N {
                augmented[row][col] -= factor * augmented[column][col];
            }
        }
    }

    Ok(augmented.into_iter().map(|row| row[N]).collect())
}

fn relative_residual(x: &[f64], rhs: &[f64]) -> f64 {
    let residual = dense_matvec(x)
        .into_iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f64>();
    let scale = rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0);
    residual / scale
}

fn max_error(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn record_solution(
    name: &'static str,
    x: &[f64],
    converged: bool,
    iterations: usize,
    reference: &[f64],
    rhs: &[f64],
) -> Result<Vec<(&'static str, f64)>, String> {
    if !converged {
        return Err(format!(
            "{name} did not converge after {iterations} iterations"
        ));
    }
    let reference_error = max_error(x, reference);
    let independent_residual = relative_residual(x, rhs);
    if !reference_error.is_finite()
        || !independent_residual.is_finite()
        || reference_error > TOLERANCE
        || independent_residual > TOLERANCE
    {
        return Err(format!(
            "{name} exceeded independent checks: error={reference_error:e}, residual={independent_residual:e}"
        ));
    }
    Ok(vec![
        (name, reference_error),
        (
            match name {
                "gmres" => "gmres_independent_residual",
                "bicgstab" => "bicgstab_independent_residual",
                "cgs" => "cgs_independent_residual",
                "idrs" => "idrs_independent_residual",
                "idrs_stab" => "idrs_stab_independent_residual",
                "fgmres" => "fgmres_independent_residual",
                _ => unreachable!("all iterative family names are listed"),
            },
            independent_residual,
        ),
    ])
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let rhs = dense_matvec(&KNOWN_SOLUTION);
    let reference = dense_reference(&rhs)?;
    let matrix = csr_matrix();
    let mut metrics = vec![(
        "reference_known_solution_error",
        max_error(&reference, &KNOWN_SOLUTION),
    )];

    let gmres = try_gmres(
        &matrix,
        &rhs,
        &GmresConfig {
            max_iterations: 200,
            restart: N,
            tolerance: TOLERANCE,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        },
    )
    .map_err(|error| format!("GMRES failed: {error:?}"))?;
    metrics.extend(record_solution(
        "gmres",
        &gmres.x,
        gmres.converged,
        gmres.iterations,
        &reference,
        &rhs,
    )?);

    let bicgstab = try_bicgstab(
        &matrix,
        &rhs,
        &BiCgstabConfig {
            max_iterations: 200,
            tolerance: TOLERANCE,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("BiCGSTAB failed: {error:?}"))?;
    metrics.extend(record_solution(
        "bicgstab",
        &bicgstab.x,
        bicgstab.converged,
        bicgstab.iterations,
        &reference,
        &rhs,
    )?);

    let cgs = try_cgs(
        &matrix,
        &rhs,
        &CgsConfig {
            max_iterations: 200,
            tolerance: TOLERANCE,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("CGS failed: {error:?}"))?;
    metrics.extend(record_solution(
        "cgs",
        &cgs.x,
        cgs.converged,
        cgs.iterations,
        &reference,
        &rhs,
    )?);

    let idrs = try_idrs(
        &matrix,
        &rhs,
        &IdrsConfig {
            s: 2,
            max_iterations: 400,
            tolerance: TOLERANCE,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("IDR(s) failed: {error:?}"))?;
    metrics.extend(record_solution(
        "idrs",
        &idrs.x,
        idrs.converged,
        idrs.iterations,
        &reference,
        &rhs,
    )?);

    let idrs_stab = try_idrs_stab(
        &matrix,
        &rhs,
        &IdrsStabConfig {
            s: 2,
            ell: 2,
            max_iterations: 400,
            tolerance: TOLERANCE,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("IDR(s)stab failed: {error:?}"))?;
    metrics.extend(record_solution(
        "idrs_stab",
        &idrs_stab.x,
        idrs_stab.converged,
        idrs_stab.iterations,
        &reference,
        &rhs,
    )?);

    let fgmres = try_fgmres(
        &matrix,
        &rhs,
        &FgmresConfig {
            max_iterations: 200,
            restart: N,
            tolerance: TOLERANCE,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        },
    )
    .map_err(|error| format!("FGMRES failed: {error:?}"))?;
    metrics.extend(record_solution(
        "fgmres",
        &fgmres.x,
        fgmres.converged,
        fgmres.iterations,
        &reference,
        &rhs,
    )?);

    let minres_matrix = diagonal_csr(&MINRES_DIAGONAL);
    let minres_rhs: Vec<f64> = MINRES_DIAGONAL
        .iter()
        .zip(MINRES_SOLUTION)
        .map(|(&diagonal, solution)| diagonal * solution)
        .collect();
    let minres = try_minres(
        &minres_matrix,
        &minres_rhs,
        &MinresConfig {
            max_iterations: MINRES_DIAGONAL.len(),
            tolerance: TOLERANCE,
            print_interval: 0,
            condition_limit: 1.0e14,
        },
    )
    .map_err(|error| format!("MINRES failed: {error:?}"))?;
    let minres_error = max_error(&minres.x, &MINRES_SOLUTION);
    let minres_residual = MINRES_DIAGONAL
        .iter()
        .zip(&minres.x)
        .zip(&minres_rhs)
        .map(|((&diagonal, &actual), &expected)| (diagonal * actual - expected).abs())
        .sum::<f64>()
        / minres_rhs
            .iter()
            .map(|value| value.abs())
            .sum::<f64>()
            .max(1.0);
    if !minres.converged
        || !minres_error.is_finite()
        || !minres_residual.is_finite()
        || minres_error > TOLERANCE
        || minres_residual > TOLERANCE
    {
        return Err(format!(
            "MINRES exceeded independent checks: converged={}, error={minres_error:e}, residual={minres_residual:e}",
            minres.converged
        ));
    }
    metrics.extend([
        ("minres_independent_solution_error", minres_error),
        ("minres_independent_residual", minres_residual),
    ]);

    Ok(metrics)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
            .reference("independent dense Gaussian-elimination and diagonal MINRES references")
                .reference_checksum("analytic/deterministic-fixture:iterative-family-reference-v1")
            .dimension("6x6 real nonsymmetric CSR plus 8x8 real symmetric-indefinite diagonal CSR systems")
        .formulation("nonsymmetric dense reference plus symmetric-indefinite diagonal reference; GMRES, FGMRES, BiCGSTAB, CGS, IDR(s), IDR(s)stab, and MINRES")
            .tolerance("all solution and independently recomputed residual errors <= 1e-10")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, TOLERANCE);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("ITERATIVE FAMILY REFERENCE FAILED");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
