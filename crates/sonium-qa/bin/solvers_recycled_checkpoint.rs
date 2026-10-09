//! Independent CPU evidence for recycled-GMRES checkpoint replay.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_qa::QaResult;
use sonium_solvers::{
    ComplexField, GmresConfig, RecycledSubspace, SolverResultContract, try_gmres_recycled,
};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.recycled-checkpoint.v1";
const TOLERANCE: f64 = 1.0e-9;

fn operator() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.5),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 5.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 5.5),
        ],
    )
}

fn scalar<T: ComplexField>(value: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_real(T::Real::from_f64(value).expect("finite scalar fixture"))
}

fn scalar_operator<T: ComplexField>() -> CsrMatrix<T>
where
    T::Real: FromPrimitive,
{
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            scalar(4.0),
            scalar(-1.0),
            scalar(-1.0),
            scalar(4.5),
            scalar(-1.0),
            scalar(-1.0),
            scalar(5.0),
            scalar(-1.0),
            scalar(-1.0),
            scalar(5.5),
        ],
    )
}

fn scalar_config<T: ComplexField>() -> GmresConfig<T::Real>
where
    T::Real: FromPrimitive,
    GmresConfig<T::Real>: Default,
{
    GmresConfig {
        max_iterations: 80,
        restart: 6,
        tolerance: T::Real::from_f64(1.0e-5).expect("finite tolerance"),
        ..GmresConfig::default()
    }
}

fn scalar_checkpoint_case<T: ComplexField>() -> Result<(f64, f64, f64), String>
where
    T::Real: FromPrimitive + ToPrimitive,
    GmresConfig<T::Real>: Default,
{
    let operator = scalar_operator::<T>();
    let first_rhs = vec![scalar(1.0), scalar(-2.0), scalar(0.5), scalar(3.0)];
    let second_rhs = vec![scalar(-0.25), scalar(1.5), scalar(2.0), scalar(-1.0)];
    let mut source = RecycledSubspace::new(3);
    let first = try_gmres_recycled(
        &operator,
        &first_rhs,
        None,
        &mut source,
        &scalar_config::<T>(),
    )
    .map_err(|error| format!("fresh scalar-family solve failed: {error}"))?;
    if !first.converged || !source.is_populated() {
        return Err("scalar-family first solve did not populate a recycled subspace".into());
    }
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("scalar-family checkpoint encode failed: {error}"))?;
    let mut restored = RecycledSubspace::<T>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("scalar-family checkpoint decode failed: {error}"))?;
    let vector_delta = restored
        .vectors()
        .iter()
        .zip(source.vectors())
        .flat_map(|(restored, original)| restored.iter().zip(original))
        .map(|(left, right)| (*left - *right).norm().to_f64().unwrap_or(f64::INFINITY))
        .fold(0.0, f64::max);

    let mut fresh_source = RecycledSubspace::new(3);
    let fresh = try_gmres_recycled(
        &operator,
        &second_rhs,
        None,
        &mut fresh_source,
        &scalar_config::<T>(),
    )
    .map_err(|error| format!("fresh scalar-family continuation failed: {error}"))?;
    let resumed = restored
        .resume(&operator, &second_rhs, None, &scalar_config::<T>())
        .map_err(|error| format!("restored scalar-family continuation failed: {error}"))?;
    if !fresh.converged || !resumed.converged {
        return Err("scalar-family continuation did not converge".into());
    }
    SolverResultContract::validate_contract(&fresh)
        .map_err(|error| format!("fresh scalar-family result contract failed: {error}"))?;
    SolverResultContract::validate_contract(&resumed)
        .map_err(|error| format!("restored scalar-family result contract failed: {error}"))?;
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

fn max_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64), String> {
    let a = operator();
    let config = GmresConfig {
        max_iterations: 80,
        restart: 6,
        tolerance: 1.0e-11,
        ..GmresConfig::default()
    };
    let first_rhs = [1.0, -2.0, 0.5, 3.0];
    let second_rhs = [-0.25, 1.5, 2.0, -1.0];
    let mut source = RecycledSubspace::new(3);
    let first = try_gmres_recycled(&a, &first_rhs, None, &mut source, &config)
        .map_err(|error| error.to_string())?;
    if !first.converged || !source.is_populated() {
        return Err("first recycled solve did not produce a populated subspace".into());
    }
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| error.to_string())?;
    let restored = RecycledSubspace::<f64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| error.to_string())?;
    let vector_error = source
        .vectors()
        .iter()
        .zip(restored.vectors())
        .map(|(a, b)| max_error(a, b))
        .fold(0.0, f64::max);

    let mut resumed = restored;
    let replay = try_gmres_recycled(&a, &second_rhs, None, &mut resumed, &config)
        .map_err(|error| error.to_string())?;
    let mut fresh = RecycledSubspace::new(3);
    let reference = try_gmres_recycled(&a, &second_rhs, None, &mut fresh, &config)
        .map_err(|error| error.to_string())?;
    if !replay.converged || !reference.converged {
        return Err("replayed or fresh recycled solve did not converge".into());
    }
    let solution_error = max_error(&replay.x, &reference.x);
    let residual = max_error(&a.matvec(&replay.x), &second_rhs);
    if vector_error > TOLERANCE || solution_error > TOLERANCE || residual > TOLERANCE {
        return Err(format!(
            "checkpoint replay exceeded tolerance: vector={vector_error:e}, solution={solution_error:e}, residual={residual:e}"
        ));
    }
    let (f32_delta, f32_residual, f32_bytes) = scalar_checkpoint_case::<f32>()?;
    let (complex32_delta, complex32_residual, complex32_bytes) =
        scalar_checkpoint_case::<Complex32>()?;
    let (complex64_delta, complex64_residual, complex64_bytes) =
        scalar_checkpoint_case::<Complex64>()?;
    Ok((
        vector_error,
        solution_error,
        residual,
        3.0,
        f32_delta.max(complex32_delta).max(complex64_delta),
        f32_residual.max(complex32_residual).max(complex64_residual),
        f32_bytes.max(complex32_bytes).max(complex64_bytes),
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            vector_error,
            solution_error,
            residual,
            scalar_family_case_count,
            scalar_family_max_delta,
            scalar_family_max_residual,
            scalar_family_max_checkpoint_bytes,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum("analytic/deterministic-fixture:recycled-checkpoint-v1")
                .reference("independent fresh recycled-GMRES replay oracle across scalar families")
                .reference_checksum("analytic/deterministic-fixture:recycled-checkpoint-v1")
                .dimension("4x4 CSR; two RHS solves with k=3 recycled vectors across f64, f32, Complex32, and Complex64")
                .formulation("portable recycled-GMRES subspace checkpoint round trip, fresh-vs-restored RHS replay, result-contract validation, and scalar-family replay")
                .tolerance("f64 checkpoint/solution/residual errors <= 1e-9; three scalar-family cases with delta/residual <= 1e-5 and non-empty packets")
                .metric("vector_round_trip_error", vector_error)
                .metric("replay_solution_error", solution_error)
                .metric("replay_residual", residual)
                .metric("scalar_family_case_count", scalar_family_case_count)
                .metric("scalar_family_max_delta", scalar_family_max_delta)
                .metric("scalar_family_max_residual", scalar_family_max_residual)
                .metric(
                    "scalar_family_max_checkpoint_bytes",
                    scalar_family_max_checkpoint_bytes,
                )
                .maximum("vector_round_trip_error", TOLERANCE)
                .maximum("replay_solution_error", TOLERANCE)
                .maximum("replay_residual", TOLERANCE)
                .minimum("scalar_family_case_count", 3.0)
                .maximum("scalar_family_max_delta", 1.0e-5)
                .maximum("scalar_family_max_residual", 1.0e-5)
                .minimum("scalar_family_max_checkpoint_bytes", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent fresh recycled-GMRES replay oracle")
                .input_checksum("analytic/deterministic-fixture:recycled-checkpoint-v1")
                .reference_checksum("analytic/deterministic-fixture:recycled-checkpoint-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
