//! Independent CPU evidence for scalar-family GMRES Krylov-basis replay.

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_qa::QaResult;
use sonium_solvers::{
    AccuracyStatus, ComplexField, CsrMatrix, GmresBasisCheckpoint, GmresBasisRun, GmresConfig,
    IdentityPreconditioner, SolverResultContract, SolverStatus, try_gmres,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.gmres-basis-checkpoint-scalars.v1";

fn matrix<T: ComplexField>() -> CsrMatrix<T>
where
    T::Real: FromPrimitive,
{
    let value = |value: f64| T::from_real(T::Real::from_f64(value).expect("scalar conversion"));
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            value(5.0),
            value(-1.0),
            value(-1.0),
            value(5.0),
            value(-1.0),
            value(-1.0),
            value(5.0),
            value(-1.0),
            value(-1.0),
            value(5.0),
        ],
    )
}

fn run_scalar<T: ComplexField>(label: &str) -> Result<(f64, f64, f64, f64, f64, f64), String>
where
    T::Real: FromPrimitive + ToPrimitive,
    GmresConfig<T::Real>: Default,
{
    let operator = matrix::<T>();
    let preconditioner = IdentityPreconditioner;
    let scalar = |value: f64| T::from_real(T::Real::from_f64(value).expect("scalar conversion"));
    let rhs = [scalar(1.0), scalar(2.0), scalar(-1.0), scalar(3.0)];
    let tolerance = T::Real::from_f64(1.0e-5).expect("tolerance conversion");
    let config = GmresConfig {
        max_iterations: 30,
        restart: 4,
        tolerance,
        ..GmresConfig::default()
    };
    let captured = GmresBasisCheckpoint::capture(&operator, &preconditioner, &rhs, &config, 2)
        .map_err(|error| format!("{label} capture failed: {error:?}"))?;
    let result_contract = if captured.solver_status() == SolverStatus::MaxIterationsReached
        && captured.accuracy_status() == AccuracyStatus::NotAttained
        && captured.validate_contract().is_ok()
    {
        1.0
    } else {
        0.0
    };
    let checkpoint = match &captured {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint.clone(),
        GmresBasisRun::Solution(_) => {
            return Err(format!("{label} fixture converged before capture"));
        }
    };
    let serialized = checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("{label} serialization failed: {error:?}"))?;
    let restored = GmresBasisCheckpoint::<T>::from_checkpoint_bytes(&serialized)
        .map_err(|error| format!("{label} restore failed: {error:?}"))?;
    let resumed = restored
        .resume(&operator, &preconditioner, &rhs, &config)
        .map_err(|error| format!("{label} resume failed: {error:?}"))?;
    let fresh = try_gmres(&operator, &rhs, &config)
        .map_err(|error| format!("{label} fresh solve failed: {error:?}"))?;
    if !resumed.converged || !fresh.converged {
        return Err(format!("{label} solve did not converge"));
    }
    let solution_error = resumed
        .x
        .iter()
        .zip(&fresh.x)
        .map(|(actual, expected)| {
            (*actual - *expected)
                .norm()
                .to_f64()
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max);
    let residual = resumed.residual.to_f64().unwrap_or(f64::INFINITY);
    let roundtrip_error = (restored.iterations() != checkpoint.iterations()
        || restored.basis_vectors() != checkpoint.basis_vectors()) as u8
        as f64;
    Ok((
        solution_error,
        residual,
        serialized.len() as f64,
        checkpoint.basis_vectors() as f64,
        roundtrip_error,
        result_contract,
    ))
}

fn run() -> Result<Vec<(&'static str, f64)>, String> {
    let (f32_solution, f32_residual, f32_bytes, f32_vectors, f32_roundtrip, f32_contract) =
        run_scalar::<f32>("f32")?;
    let (c32_solution, c32_residual, c32_bytes, c32_vectors, c32_roundtrip, c32_contract) =
        run_scalar::<Complex32>("Complex32")?;
    let (c64_solution, c64_residual, c64_bytes, c64_vectors, c64_roundtrip, c64_contract) =
        run_scalar::<Complex64>("Complex64")?;

    let f32_checkpoint = GmresBasisCheckpoint::capture(
        &matrix::<f32>(),
        &IdentityPreconditioner,
        &[1.0_f32, 2.0, -1.0, 3.0],
        &GmresConfig {
            max_iterations: 30,
            restart: 4,
            tolerance: 1.0e-5_f32,
            ..GmresConfig::default()
        },
        2,
    )
    .map_err(|error| format!("f32 mismatch fixture failed: {error:?}"))?;
    let f32_checkpoint = match f32_checkpoint {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => return Err("f32 mismatch fixture converged early".into()),
    };
    let f32_mismatch_bytes = f32_checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("f32 mismatch serialization failed: {error:?}"))?;
    let f32_restore_rejected =
        GmresBasisCheckpoint::<f64>::from_checkpoint_bytes(&f32_mismatch_bytes).is_err();

    let c32_rhs = [
        Complex32::new(1.0, 0.25),
        Complex32::new(2.0, -0.5),
        Complex32::new(-1.0, 0.75),
        Complex32::new(3.0, -0.25),
    ];
    let c32_config = GmresConfig {
        max_iterations: 30,
        restart: 4,
        tolerance: 1.0e-5_f32,
        ..GmresConfig::default()
    };
    let c32_checkpoint = GmresBasisCheckpoint::capture(
        &matrix::<Complex32>(),
        &IdentityPreconditioner,
        &c32_rhs,
        &c32_config,
        2,
    )
    .map_err(|error| format!("Complex32 mismatch fixture failed: {error:?}"))?;
    let c32_checkpoint = match c32_checkpoint {
        GmresBasisRun::Checkpoint(checkpoint) => checkpoint,
        GmresBasisRun::Solution(_) => {
            return Err("Complex32 mismatch fixture converged early".into());
        }
    };
    let c32_mismatch_bytes = c32_checkpoint
        .to_checkpoint_bytes()
        .map_err(|error| format!("Complex32 mismatch serialization failed: {error:?}"))?;
    let c32_restore_rejected =
        GmresBasisCheckpoint::<Complex64>::from_checkpoint_bytes(&c32_mismatch_bytes).is_err();

    Ok(vec![
        ("f32_solution_error", f32_solution),
        ("f32_residual", f32_residual),
        ("f32_serialized_bytes", f32_bytes),
        ("f32_basis_vectors", f32_vectors),
        ("f32_roundtrip_shape_error", f32_roundtrip),
        ("f32_result_contract", f32_contract),
        ("complex32_solution_error", c32_solution),
        ("complex32_residual", c32_residual),
        ("complex32_serialized_bytes", c32_bytes),
        ("complex32_basis_vectors", c32_vectors),
        ("complex32_roundtrip_shape_error", c32_roundtrip),
        ("complex32_result_contract", c32_contract),
        ("complex64_solution_error", c64_solution),
        ("complex64_residual", c64_residual),
        ("complex64_serialized_bytes", c64_bytes),
        ("complex64_basis_vectors", c64_vectors),
        ("complex64_roundtrip_shape_error", c64_roundtrip),
        ("complex64_result_contract", c64_contract),
        (
            "cross_family_rejection_checks",
            f32_restore_rejected as u8 as f64 + c32_restore_rejected as u8 as f64,
        ),
    ])
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let mut result = QaResult::passed(CASE_ID)
                .reference("analytic 4x4 tridiagonal SPD matrix; independent scalar-family replay")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-scalars-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-basis-checkpoint-scalars-v1",
                )
                .dimension("4x4 CSR; fixed-restart GMRES; f32, Complex32, and Complex64")
                .tolerance("solution/residual errors <= 1e-3; scalar-family mismatch checks = 2")
                .seed("solvers-gmres-basis-checkpoint-scalars:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            for (name, value) in metrics {
                if name.contains("error")
                    || name == "f32_residual"
                    || name == "complex32_residual"
                    || name == "complex64_residual"
                {
                    result = result.metric(name, value).maximum(name, 1.0e-3);
                } else if name == "cross_family_rejection_checks" {
                    result = result.metric(name, value).minimum(name, 2.0);
                } else {
                    result = result.metric(name, value);
                }
            }
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("analytic scalar-family GMRES Krylov-basis replay")
                .input_checksum("analytic/deterministic-fixture:gmres-basis-checkpoint-scalars-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:gmres-basis-checkpoint-scalars-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("FAILED: {error}");
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
