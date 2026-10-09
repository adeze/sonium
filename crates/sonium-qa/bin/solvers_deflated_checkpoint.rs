//! Independent CPU evidence for portable deflated-GMRES subspace state.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_qa::QaResult;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{
    ComplexField, DeflatedCheckpoint, DeflationSubspace, DiagonalPreconditioner, GmresConfig,
    SolverResultContract, try_gmres_deflated, try_gmres_deflated_preconditioned,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.deflated-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-10;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_dense(
        &sonium_solvers::array![[4.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        1.0e-15,
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 20,
        restart: 3,
        tolerance: 1.0e-12,
        print_interval: 0,
        ..Default::default()
    }
}

fn scalar<T: ComplexField>(value: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_real(T::Real::from_f64(value).expect("finite scalar fixture"))
}

fn scalar_matrix<T: ComplexField>() -> CsrMatrix<T>
where
    T::Real: FromPrimitive,
{
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![scalar(4.0), scalar(3.0), scalar(2.0)],
    )
}

fn scalar_config<T: ComplexField>() -> GmresConfig<T::Real>
where
    T::Real: FromPrimitive,
    GmresConfig<T::Real>: Default,
{
    GmresConfig {
        max_iterations: 20,
        restart: 3,
        tolerance: T::Real::from_f64(1.0e-5).expect("finite tolerance"),
        print_interval: 0,
        ..Default::default()
    }
}

fn scalar_checkpoint_case<T: ComplexField>() -> Result<(f64, f64, f64), String>
where
    T::Real: FromPrimitive + ToPrimitive,
    GmresConfig<T::Real>: Default,
{
    let operator = scalar_matrix::<T>();
    let deflation = DeflationSubspace::try_new(
        vec![vec![scalar(0.7), scalar(-0.3), scalar(0.5)]],
        &operator,
    )
    .map_err(|error| format!("deflation setup failed: {error}"))?;
    let checkpoint = DeflatedCheckpoint::from_subspace(&deflation)
        .map_err(|error| format!("checkpoint capture failed: {error}"))?;
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint encode failed: {error}"))?;
    let restored = DeflatedCheckpoint::<T>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("checkpoint decode failed: {error}"))?;
    let vector_delta = restored
        .vectors()
        .iter()
        .zip(checkpoint.vectors())
        .flat_map(|(restored, original)| restored.iter().zip(original))
        .map(|(left, right)| (*left - *right).norm().to_f64().unwrap_or(f64::INFINITY))
        .fold(0.0, f64::max);
    let rhs = vec![scalar(1.0), scalar(-2.0), scalar(3.0)];
    let fresh = try_gmres_deflated(&operator, &deflation, &rhs, None, &scalar_config::<T>())
        .map_err(|error| format!("fresh deflated solve failed: {error}"))?;
    let resumed = restored
        .resume(&operator, &rhs, None, &scalar_config::<T>())
        .map_err(|error| format!("restored deflated solve failed: {error}"))?;
    if !fresh.converged || !resumed.converged {
        return Err("scalar-family deflated solve did not converge".to_string());
    }
    SolverResultContract::validate_contract(&fresh)
        .map_err(|error| format!("fresh result contract failed: {error}"))?;
    SolverResultContract::validate_contract(&resumed)
        .map_err(|error| format!("restored result contract failed: {error}"))?;
    let solution_delta = fresh
        .x
        .iter()
        .zip(&resumed.x)
        .map(|(left, right)| (*left - *right).norm().to_f64().unwrap_or(f64::INFINITY))
        .fold(0.0, f64::max);
    let residual = resumed.residual.to_f64().unwrap_or(f64::INFINITY);
    Ok((
        solution_delta.max(vector_delta),
        residual,
        bytes.len() as f64,
    ))
}

fn relative_residual(matrix: &CsrMatrix<f64>, solution: &[f64], rhs: &[f64]) -> f64 {
    let action = matrix.matvec(solution);
    let residual = action
        .iter()
        .zip(rhs)
        .map(|(value, target)| (value - target).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    residual / rhs_norm.max(f64::MIN_POSITIVE)
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let matrix = matrix();
    let deflation = DeflationSubspace::try_new(vec![vec![1.0, 0.0, 0.0]], &matrix)
        .map_err(|error| format!("deflation setup failed: {error}"))?;
    let checkpoint = DeflatedCheckpoint::from_subspace(&deflation)
        .map_err(|error| format!("checkpoint capture failed: {error}"))?;
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint serialization failed: {error}"))?;
    let restored = DeflatedCheckpoint::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("checkpoint restore failed: {error}"))?;
    let empty = DeflatedCheckpoint::<f64>::from_vectors(&[])
        .map_err(|error| format!("empty checkpoint capture failed: {error}"))?;
    let mut malformed_empty = empty
        .to_checkpoint_bytes()
        .map_err(|error| format!("empty checkpoint serialization failed: {error}"))?;
    malformed_empty[14..22].copy_from_slice(&1_u64.to_le_bytes());
    let malformed_header_rejected =
        DeflatedCheckpoint::<f64>::from_checkpoint_bytes(&malformed_empty).is_err();
    if !malformed_header_rejected {
        return Err("empty deflated checkpoint accepted nonzero dimension header".to_string());
    }
    let rhs = [1.0, -2.0, 3.0];
    let fresh = try_gmres_deflated(&matrix, &deflation, &rhs, None, &config())
        .map_err(|error| format!("fresh solve failed: {error}"))?;
    let resumed = restored
        .resume(&matrix, &rhs, None, &config())
        .map_err(|error| format!("resumed solve failed: {error}"))?;
    if !fresh.converged || !resumed.converged {
        return Err("fresh or resumed solve did not converge".to_string());
    }
    let solution_error = fresh
        .x
        .iter()
        .zip(&resumed.x)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    let residual = relative_residual(&matrix, &resumed.x, &rhs);
    let preconditioner = DiagonalPreconditioner::from_csr(&matrix);
    let expected = [0.25, -2.0 / 3.0, 1.5];
    let preconditioned_fresh = try_gmres_deflated_preconditioned(
        &matrix,
        &preconditioner,
        &deflation,
        &rhs,
        None,
        &config(),
    )
    .map_err(|error| format!("fresh preconditioned solve failed: {error}"))?;
    let preconditioned_resumed = restored
        .resume_preconditioned(&matrix, &preconditioner, &rhs, Some(&expected), &config())
        .map_err(|error| format!("resumed preconditioned solve failed: {error}"))?;
    if !preconditioned_fresh.converged
        || !preconditioned_resumed.converged
        || preconditioned_resumed.iterations != 0
    {
        return Err(format!(
            "preconditioned continuation contract failed: fresh_converged={} resumed_converged={} resumed_iterations={}",
            preconditioned_fresh.converged,
            preconditioned_resumed.converged,
            preconditioned_resumed.iterations,
        ));
    }
    let malformed_guess_rejected = restored
        .resume_preconditioned(&matrix, &preconditioner, &rhs, Some(&[0.0, 0.0]), &config())
        .is_err();
    if !malformed_guess_rejected {
        return Err("preconditioned malformed guess was accepted".to_string());
    }
    let preconditioned_solution_error = preconditioned_fresh
        .x
        .iter()
        .zip(expected)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0, f64::max);
    let preconditioned_residual = relative_residual(&matrix, &preconditioned_resumed.x, &rhs);
    let (f32_solution_delta, f32_residual, f32_bytes) = scalar_checkpoint_case::<f32>()?;
    let (complex32_solution_delta, complex32_residual, complex32_bytes) =
        scalar_checkpoint_case::<Complex32>()?;
    let (complex64_solution_delta, complex64_residual, complex64_bytes) =
        scalar_checkpoint_case::<Complex64>()?;
    let scalar_max_solution_delta = f32_solution_delta
        .max(complex32_solution_delta)
        .max(complex64_solution_delta);
    let scalar_max_residual = f32_residual.max(complex32_residual).max(complex64_residual);
    let scalar_max_checkpoint_bytes = f32_bytes.max(complex32_bytes).max(complex64_bytes);
    Ok(vec![
        ("checkpoint_bytes", bytes.len() as f64),
        ("solution_error", solution_error),
        ("residual", residual),
        (
            "preconditioned_solution_error",
            preconditioned_solution_error,
        ),
        ("preconditioned_residual", preconditioned_residual),
        (
            "preconditioned_resumed_iterations",
            preconditioned_resumed.iterations as f64,
        ),
        ("preconditioned_continuation_contract", 1.0),
        (
            "preconditioned_malformed_guess_rejection",
            if malformed_guess_rejected { 1.0 } else { 0.0 },
        ),
        (
            "malformed_empty_dimension_rejection",
            if malformed_header_rejected { 1.0 } else { 0.0 },
        ),
        ("scalar_family_case_count", 3.0),
        (
            "scalar_family_max_solution_delta",
            scalar_max_solution_delta,
        ),
        ("scalar_family_max_residual", scalar_max_residual),
        (
            "scalar_family_max_checkpoint_bytes",
            scalar_max_checkpoint_bytes,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic diagonal 3x3 CSR deflated-GMRES continuation")
                .input_checksum("analytic/deterministic-fixture:deflated-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:deflated-checkpoint-v1")
                .dimension("3x3 CSR; one deflation vector across f64, f32, Complex32, and Complex64")
                .formulation("portable deflated-GMRES vector checkpoint restore/resume, preconditioned continuation, malformed-header/guess rejection, and scalar-family replay")
                .tolerance("f64 solution/residual errors <= 1e-10; scalar-family solution delta/residual <= 1e-5; three scalar-family cases and non-empty packets")
                .seed("solvers-deflated-checkpoint:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                result = match name {
                    "checkpoint_bytes" => result.metric(name, value),
                    "scalar_family_case_count" => result.minimum(name, 3.0).metric(name, value),
                    "scalar_family_max_checkpoint_bytes" => {
                        result.minimum(name, 1.0).metric(name, value)
                    }
                    "scalar_family_max_solution_delta" | "scalar_family_max_residual" => {
                        result.maximum(name, 1.0e-5).metric(name, value)
                    }
                    "preconditioned_continuation_contract"
                    | "preconditioned_malformed_guess_rejection"
                    | "malformed_empty_dimension_rejection" => {
                        result.minimum(name, 1.0).metric(name, value)
                    }
                    _ => result.maximum(name, TOLERANCE).metric(name, value),
                };
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .reference("analytic diagonal 3x3 CSR deflated-GMRES continuation")
                .input_checksum("analytic/deterministic-fixture:deflated-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:deflated-checkpoint-v1")
                .dimension("3x3 CSR; one deflation vector across f64, f32, Complex32, and Complex64")
                .formulation("portable deflated-GMRES vector checkpoint restore/resume, preconditioned continuation, malformed-header/guess rejection, and scalar-family replay")
                .tolerance("f64 solution/residual errors <= 1e-10; scalar-family solution delta/residual <= 1e-5; three scalar-family cases and non-empty packets")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
