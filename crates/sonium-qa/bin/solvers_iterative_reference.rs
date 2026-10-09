//! Independent CPU reference for nonsymmetric iterative solver results.

use sonium_qa::QaResult;
use sonium_solvers::{BiCgstabConfig, CsrBuilder, GmresConfig, bicgstab, gmres};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-independent-reference.v1";
const N: usize = 6;

const MATRIX: [[f64; N]; N] = [
    [5.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [2.0, 6.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 5.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 7.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 4.0],
];

const KNOWN_SOLUTION: [f64; N] = [1.0, -0.5, 0.75, 1.5, -1.25, 0.25];

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

/// Small independent dense reference with partial pivoting.
fn dense_reference(rhs: &[f64]) -> Result<Vec<f64>, String> {
    let mut augmented = vec![vec![0.0; N + 1]; N];
    for row in 0..N {
        augmented[row][..N].copy_from_slice(&MATRIX[row]);
        augmented[row][N] = rhs[row];
    }

    for pivot in 0..N {
        let (pivot_row, pivot_value) = (pivot..N)
            .map(|row| (row, augmented[row][pivot].abs()))
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| "empty pivot search".to_string())?;
        if pivot_value <= 1.0e-14 {
            return Err(format!("singular reference pivot {pivot}"));
        }
        augmented.swap(pivot, pivot_row);
        for row in (pivot + 1)..N {
            let factor = augmented[row][pivot] / augmented[pivot][pivot];
            for column in pivot..=N {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }

    let mut solution = vec![0.0; N];
    for row in (0..N).rev() {
        let tail: f64 = ((row + 1)..N)
            .map(|column| augmented[row][column] * solution[column])
            .sum();
        solution[row] = (augmented[row][N] - tail) / augmented[row][row];
    }
    Ok(solution)
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

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let rhs = dense_matvec(&KNOWN_SOLUTION);
    let reference = dense_reference(&rhs)?;
    let matrix = csr_matrix();

    let gmres = gmres(
        &matrix,
        &rhs,
        &GmresConfig {
            max_iterations: 100,
            restart: N,
            tolerance: 1.0e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        },
    );
    let bicgstab = bicgstab(
        &matrix,
        &rhs,
        &BiCgstabConfig {
            max_iterations: 100,
            tolerance: 1.0e-12,
            print_interval: 0,
        },
    );

    if !gmres.converged || !bicgstab.converged {
        return Err(format!(
            "iterative solver did not converge: GMRES={}, BiCGSTAB={}",
            gmres.converged, bicgstab.converged
        ));
    }

    let reference_known_error = max_error(&reference, &KNOWN_SOLUTION);
    let gmres_error = max_error(&gmres.x, &reference);
    let bicgstab_error = max_error(&bicgstab.x, &reference);
    let gmres_residual = relative_residual(&gmres.x, &rhs);
    let bicgstab_residual = relative_residual(&bicgstab.x, &rhs);
    let values = [
        ("reference_known_solution_error", reference_known_error),
        ("gmres_reference_error", gmres_error),
        ("bicgstab_reference_error", bicgstab_error),
        ("gmres_independent_residual", gmres_residual),
        ("bicgstab_independent_residual", bicgstab_residual),
    ];
    if values.iter().any(|(_, value)| !value.is_finite()) {
        return Err("independent reference produced a non-finite metric".into());
    }
    Ok(values.into_iter().collect())
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent dense Gaussian-elimination reference")
                .reference_checksum("analytic/deterministic-fixture:iterative-reference-v1")
                .dimension("6x6 real nonsymmetric diagonally dominant CSR system")
                .formulation("GMRES and BiCGSTAB against an independent dense reference")
                .tolerance("reference and independent residual errors <= 1e-10")
                .timing_ms(
                    "reference_and_iterative_solves",
                    started.elapsed().as_millis() as u64,
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, 1.0e-10);
            }
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("ITERATIVE REFERENCE FAILED");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
