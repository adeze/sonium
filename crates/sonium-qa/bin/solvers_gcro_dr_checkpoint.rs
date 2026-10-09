//! Independent CPU evidence for portable GCRO-DR recycled-subspace state.

use sonium_qa::QaResult;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{GcroDrCheckpoint, GcroDrConfig, GcroDrSubspace, try_gcro_dr};
use std::time::Instant;

const CASE_ID: &str = "solvers.gcro-dr-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-9;

fn matrix() -> CsrMatrix<f64> {
    let n = 12;
    let mut triplets = Vec::new();
    for row in 0..n {
        triplets.push((row, row, 4.0));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn config() -> GcroDrConfig<f64> {
    GcroDrConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-11,
        print_interval: 0,
    }
}

fn relative_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    let action = matrix.matvec(solution);
    let numerator = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = rhs.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
    numerator / denominator.max(1.0)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let expected_first: Vec<f64> = (0..12)
        .map(|index| 0.2 + 0.03 * index as f64 - 0.001 * (index * index) as f64)
        .collect();
    let expected_second: Vec<f64> = (0..12)
        .map(|index| -0.4 + 0.02 * index as f64 + 0.0007 * (index * index) as f64)
        .collect();
    let first_rhs = matrix.matvec(&expected_first);
    let second_rhs = matrix.matvec(&expected_second);
    let mut first_subspace = GcroDrSubspace::new(2);
    let first = try_gcro_dr(&matrix, &first_rhs, None, &mut first_subspace, &config())
        .map_err(|error| format!("first GCRO-DR solve failed: {error:?}"))?;
    if !first.converged || first_subspace.num_vectors() == 0 {
        return Err(format!(
            "first solve did not produce a converged recycled state: status={:?}, vectors={}",
            first.status,
            first_subspace.num_vectors()
        ));
    }
    let checkpoint = GcroDrCheckpoint::from_subspace(&first_subspace)
        .map_err(|error| format!("checkpoint capture failed: {error:?}"))?;
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint serialization failed: {error:?}"))?;
    let restored = GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("checkpoint restore failed: {error:?}"))?;
    let empty = GcroDrCheckpoint::<f64>::from_vectors(1, &[])
        .map_err(|error| format!("empty checkpoint capture failed: {error:?}"))?;
    let mut malformed_empty = empty
        .to_checkpoint_bytes()
        .map_err(|error| format!("empty checkpoint serialization failed: {error:?}"))?;
    malformed_empty[14..22].copy_from_slice(&1_u64.to_le_bytes());
    let malformed_header_rejected =
        GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&malformed_empty).is_err();
    if !malformed_header_rejected {
        return Err("empty GCRO-DR checkpoint accepted nonzero dimension header".to_string());
    }
    let (updated_subspace, resumed) = restored
        .resume(&matrix, &second_rhs, None, &config())
        .map_err(|error| format!("checkpoint resume failed: {error:?}"))?;
    if !resumed.converged || updated_subspace.num_vectors() > updated_subspace.target_size() {
        return Err(format!(
            "resumed GCRO-DR state invalid: status={:?}, vectors={}/{}",
            resumed.status,
            updated_subspace.num_vectors(),
            updated_subspace.target_size()
        ));
    }
    let solution_error = resumed
        .x
        .iter()
        .zip(&expected_second)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let independent_residual = relative_residual(&matrix, &resumed.x, &second_rhs);
    if solution_error > TOLERANCE || independent_residual > TOLERANCE {
        return Err(format!(
            "GCRO-DR checkpoint errors exceed tolerance: solution={solution_error:.3e}, residual={independent_residual:.3e}"
        ));
    }
    Ok(vec![
        ("solution_error", solution_error),
        ("independent_residual", independent_residual),
        ("recycled_vectors", restored.num_vectors() as f64),
        ("checkpoint_bytes", bytes.len() as f64),
        ("source_iterations", first.iterations as f64),
        ("resumed_iterations", resumed.iterations as f64),
        (
            "malformed_empty_dimension_rejection",
            if malformed_header_rejected { 1.0 } else { 0.0 },
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent analytic 12x12 tridiagonal GCRO-DR reference")
                .input_checksum("analytic/deterministic-fixture:gcro-dr-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:gcro-dr-checkpoint-v1")
                .dimension("12x12 CSR; GCRO-DR restart=3, recycled target=2")
                .tolerance("solution error and independent residual <= 1e-9")
                .seed("solvers-gcro-dr-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE).metric(name, value);
                } else {
                    result = result.metric(name, value);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent analytic 12x12 tridiagonal GCRO-DR reference")
                .input_checksum("analytic/deterministic-fixture:gcro-dr-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:gcro-dr-checkpoint-v1")
                .tolerance("solution error and independent residual <= 1e-9")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
