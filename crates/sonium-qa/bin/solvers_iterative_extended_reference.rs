//! Independent CPU references for the remaining public iterative families.

use sonium_qa::QaResult;
use sonium_solvers::{CaCgBasis, CaCgConfig, CsrBuilder, TfqmrConfig, try_ca_cg, try_tfqmr};
use std::time::Instant;

const CASE_ID: &str = "solvers.iterative-extended-reference.v1";
const TOLERANCE: f64 = 1.0e-10;

const NONSYMMETRIC: [[f64; 5]; 5] = [
    [5.0, 1.0, 0.0, 0.0, 0.0],
    [2.0, 6.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, 2.0, 0.0],
    [0.0, 0.0, 1.0, 5.0, 1.0],
    [1.0, 0.0, 0.0, -2.0, 7.0],
];
const NONSYMMETRIC_X: [f64; 5] = [1.0, -0.5, 0.75, 1.5, -1.25];

const SPD: [[f64; 4]; 4] = [
    [6.0, 1.0, 0.0, 0.0],
    [1.0, 5.0, 1.0, 0.0],
    [0.0, 1.0, 4.0, 1.0],
    [0.0, 0.0, 1.0, 3.0],
];
const SPD_X: [f64; 4] = [1.0, -2.0, 0.5, 1.25];

fn matvec<const N: usize>(matrix: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|row| matrix[row].iter().zip(x).map(|(a, x)| a * x).sum())
}

fn csr<const N: usize>(matrix: &[[f64; N]; N]) -> sonium_solvers::CsrMatrix<f64> {
    let mut builder = CsrBuilder::new(N, N);
    for row in matrix {
        builder.add_row_entries(
            row.iter()
                .enumerate()
                .filter(|(_, value)| value.abs() > f64::EPSILON)
                .map(|(column, &value)| (column, value)),
        );
    }
    builder.finish()
}

fn dense_reference<const N: usize>(matrix: &[[f64; N]; N], rhs: &[f64; N]) -> [f64; N] {
    let mut augmented = vec![vec![0.0; N + 1]; N];
    for row in 0..N {
        augmented[row][..N].copy_from_slice(&matrix[row]);
        augmented[row][N] = rhs[row];
    }
    for column in 0..N {
        let pivot = (column..N)
            .max_by(|&left, &right| {
                augmented[left][column]
                    .abs()
                    .total_cmp(&augmented[right][column].abs())
            })
            .expect("non-empty pivot range");
        assert!(augmented[pivot][column].abs() > 1.0e-14);
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
            for index in column..=N {
                augmented[row][index] -= factor * augmented[column][index];
            }
        }
    }
    std::array::from_fn(|row| augmented[row][N])
}

fn max_error<const N: usize>(actual: &[f64], expected: &[f64; N]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn relative_residual<const N: usize>(matrix: &[[f64; N]; N], x: &[f64], rhs: &[f64; N]) -> f64 {
    let mut residual = 0.0;
    for row in 0..N {
        let value: f64 = matrix[row].iter().zip(x).map(|(a, x)| a * x).sum();
        residual += (value - rhs[row]).abs();
    }
    residual / rhs.iter().map(|value| value.abs()).sum::<f64>().max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let nonsymmetric_rhs = matvec(&NONSYMMETRIC, &NONSYMMETRIC_X);
    let nonsymmetric_reference = dense_reference(&NONSYMMETRIC, &nonsymmetric_rhs);
    let tfqmr = try_tfqmr(
        &csr(&NONSYMMETRIC),
        &nonsymmetric_rhs,
        &TfqmrConfig {
            max_iterations: 400,
            tolerance: TOLERANCE,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("TFQMR failed: {error:?}"))?;
    if !tfqmr.converged {
        return Err(format!("TFQMR did not converge: {:?}", tfqmr.status));
    }

    let spd_rhs = matvec(&SPD, &SPD_X);
    let spd_reference = dense_reference(&SPD, &spd_rhs);
    let ca_cg = try_ca_cg(
        &csr(&SPD),
        &spd_rhs,
        &CaCgConfig {
            max_iterations: 100,
            tolerance: TOLERANCE,
            step_size: 3,
            basis: CaCgBasis::Monomial,
            print_interval: 0,
        },
    )
    .map_err(|error| format!("CA-CG failed: {error:?}"))?;
    if !ca_cg.converged {
        return Err(format!("CA-CG did not converge: {:?}", ca_cg.status));
    }

    let metrics = [
        (
            "tfqmr_reference_error",
            max_error(&tfqmr.x, &nonsymmetric_reference),
        ),
        (
            "tfqmr_independent_residual",
            relative_residual(&NONSYMMETRIC, &tfqmr.x, &nonsymmetric_rhs),
        ),
        ("ca_cg_reference_error", max_error(&ca_cg.x, &spd_reference)),
        (
            "ca_cg_independent_residual",
            relative_residual(&SPD, &ca_cg.x, &spd_rhs),
        ),
    ];
    for (name, value) in metrics {
        if !value.is_finite() || value > TOLERANCE {
            return Err(format!("{name}={value:e} exceeds {TOLERANCE:e}"));
        }
    }
    Ok(metrics.into_iter().collect())
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent dense partial-pivot references for TFQMR and CA-CG")
                .reference_checksum(
                    "analytic/deterministic-fixture:iterative-extended-reference-v1",
                )
                .input_checksum("analytic/deterministic-fixture:iterative-extended-reference-v1")
                .dimension("5x5 real nonsymmetric TFQMR plus 4x4 real SPD CA-CG")
                .formulation("production checked TFQMR and communication-avoiding CG")
                .tolerance("all independently recomputed errors and residuals <= 1e-10")
                .seed("solvers-iterative-extended-reference:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = result.metric(name, value).maximum(name, TOLERANCE);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent dense partial-pivot references for TFQMR and CA-CG")
                .reference_checksum(
                    "analytic/deterministic-fixture:iterative-extended-reference-v1",
                )
                .input_checksum("analytic/deterministic-fixture:iterative-extended-reference-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("ITERATIVE EXTENDED REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
