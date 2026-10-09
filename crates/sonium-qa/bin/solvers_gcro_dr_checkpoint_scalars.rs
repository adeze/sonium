//! Independent CPU evidence for GCRO-DR checkpoint scalar-family replay.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_qa::QaResult;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{
    ComplexField, GcroDrCheckpoint, GcroDrConfig, GcroDrSubspace, SolverResultContract, try_gcro_dr,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.gcro-dr-checkpoint-scalars.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:gcro-dr-checkpoint-scalars-v1";
const TOLERANCE: f64 = 1.0e-4;

fn value<T: ComplexField>(real: f64, imag: f64) -> T {
    T::from_re_im(
        T::Real::from_f64(real).expect("real fixture value should convert"),
        T::Real::from_f64(imag).expect("imaginary fixture value should convert"),
    )
}

fn matrix<T: ComplexField>() -> CsrMatrix<T> {
    let n = 12;
    let mut triplets = Vec::new();
    for row in 0..n {
        triplets.push((row, row, value::<T>(4.0, 0.0)));
        if row > 0 {
            triplets.push((row, row - 1, value::<T>(-1.0, 0.0)));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, value::<T>(-1.0, 0.0)));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn config<T: ComplexField>() -> GcroDrConfig<T::Real> {
    let tolerance = if std::mem::size_of::<T::Real>() == 4 {
        1.0e-5
    } else {
        1.0e-11
    };
    GcroDrConfig {
        max_iterations: 60,
        restart: 4,
        tolerance: T::Real::from_f64(tolerance).unwrap(),
        print_interval: 0,
    }
}

fn residual<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], rhs: &[T]) -> f64 {
    let action = matrix.matvec(x);
    let numerator = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr().to_f64().unwrap())
        .sum::<f64>()
        .sqrt();
    let denominator = rhs
        .iter()
        .map(|value| value.norm_sqr().to_f64().unwrap())
        .sum::<f64>()
        .sqrt();
    numerator / denominator.max(1.0)
}

fn family_case<T: ComplexField>() -> Result<(f64, f64, f64, f64, f64), String> {
    let operator = matrix::<T>();
    let expected_first: Vec<T> = (0..12)
        .map(|index| {
            value(
                0.2 + 0.03 * index as f64 - 0.001 * (index * index) as f64,
                0.1 - 0.004 * index as f64,
            )
        })
        .collect();
    let expected_second: Vec<T> = (0..12)
        .map(|index| {
            value(
                -0.4 + 0.02 * index as f64 + 0.0007 * (index * index) as f64,
                -0.08 + 0.006 * index as f64,
            )
        })
        .collect();
    let first_rhs = operator.matvec(&expected_first);
    let second_rhs = operator.matvec(&expected_second);
    let config = config::<T>();

    let mut first_subspace = GcroDrSubspace::new(2);
    let first = try_gcro_dr(&operator, &first_rhs, None, &mut first_subspace, &config)
        .map_err(|error| format!("GCRO-DR source solve failed: {error}"))?;
    if !first.converged {
        return Err(format!(
            "GCRO-DR source solve did not converge: {:?}",
            first.status
        ));
    }
    let checkpoint = GcroDrCheckpoint::from_subspace(&first_subspace)
        .map_err(|error| format!("GCRO-DR checkpoint capture failed: {error}"))?;
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("GCRO-DR checkpoint encode failed: {error}"))?;
    let restored = GcroDrCheckpoint::<T>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("GCRO-DR checkpoint decode failed: {error}"))?;

    let (updated_subspace, resumed) = restored
        .resume(&operator, &second_rhs, None, &config)
        .map_err(|error| format!("GCRO-DR checkpoint resume failed: {error}"))?;
    if !resumed.converged {
        return Err(format!(
            "GCRO-DR resumed solve did not converge: {:?}",
            resumed.status
        ));
    }
    resumed
        .validate_contract()
        .map_err(|error| format!("GCRO-DR resumed result contract failed: {error}"))?;

    let mut fresh_subspace = GcroDrSubspace::new(2);
    let fresh = try_gcro_dr(&operator, &second_rhs, None, &mut fresh_subspace, &config)
        .map_err(|error| format!("GCRO-DR fresh solve failed: {error}"))?;
    let solution_delta = resumed
        .x
        .iter()
        .zip(&fresh.x)
        .map(|(actual, expected)| (*actual - *expected).norm().to_f64().unwrap())
        .fold(0.0, f64::max);
    let independent_residual = residual(&operator, &resumed.x, &second_rhs);
    if solution_delta > TOLERANCE || independent_residual > TOLERANCE {
        return Err(format!(
            "GCRO-DR scalar-family error exceeded tolerance: delta={solution_delta:e}, residual={independent_residual:e}"
        ));
    }
    Ok((
        solution_delta,
        independent_residual,
        bytes.len() as f64,
        updated_subspace.num_vectors() as f64,
        resumed.iterations as f64,
    ))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let f32_case = family_case::<f32>()?;
    let complex32_case = family_case::<Complex32>()?;
    let complex64_case = family_case::<Complex64>()?;
    let cases = [f32_case, complex32_case, complex64_case];

    let cross_family_source = GcroDrCheckpoint::<f32>::from_vectors(2, &[vec![1.0]])
        .map_err(|error| format!("cross-family source checkpoint failed: {error}"))?
        .to_checkpoint_bytes()
        .map_err(|error| format!("cross-family source encode failed: {error}"))?;
    let cross_family_rejection =
        GcroDrCheckpoint::<Complex32>::from_checkpoint_bytes(&cross_family_source).is_err();
    if !cross_family_rejection {
        return Err("GCRO-DR accepted an f32 checkpoint as Complex32".into());
    }

    Ok(vec![
        ("scalar_family_case_count", cases.len() as f64),
        (
            "scalar_family_max_solution_delta",
            cases.iter().map(|case| case.0).fold(0.0, f64::max),
        ),
        (
            "scalar_family_max_residual",
            cases.iter().map(|case| case.1).fold(0.0, f64::max),
        ),
        (
            "scalar_family_max_checkpoint_bytes",
            cases.iter().map(|case| case.2).fold(0.0, f64::max),
        ),
        (
            "scalar_family_max_recycled_vectors",
            cases.iter().map(|case| case.3).fold(0.0, f64::max),
        ),
        (
            "scalar_family_max_resumed_iterations",
            cases.iter().map(|case| case.4).fold(0.0, f64::max),
        ),
        (
            "cross_family_rejection",
            if cross_family_rejection { 1.0 } else { 0.0 },
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("independent GCRO-DR checkpoint replay across real and complex scalar families")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .formulation("analytic 12x12 tridiagonal CSR with two sequential recycled-subspace solves")
                .dimension("12x12 CSR; f32, Complex32, Complex64; GCRO-DR restart=4, target=2")
                .tolerance("three scalar-family cases; solution delta and independent residual <= 1e-4; cross-family payload rejects")
                .seed("solvers-gcro-dr-checkpoint-scalars:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("delta") || name.contains("residual") {
                    result = result.maximum(name, TOLERANCE);
                }
                result = result.metric(name, value);
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
