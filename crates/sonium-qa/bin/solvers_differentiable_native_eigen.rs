//! Independent CPU evidence for native eigen states in the differentiable contract.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::eigen::{ComplexTargetedEigenResult, TargetedEigenResult};
use sonium_solvers::traits::SolverError;
use sonium_solvers::{
    ClosureDifferentiableSolve, ContractValidation, DifferentiableContractValue,
    DifferentiableSolveContract, DifferentiableSystemKind,
};
use sonium_solvers_core::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.differentiable-native-eigen.v1";

fn real_result() -> TargetedEigenResult {
    TargetedEigenResult {
        eigenvalues: vec![1.0, 2.0],
        eigenvectors: DenseMatrix::from_row_major_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        residual_norms: vec![1.0e-10, 2.0e-10],
        iterations: 2,
        converged: true,
    }
}

fn complex_result() -> ComplexTargetedEigenResult {
    ComplexTargetedEigenResult {
        eigenvalues: vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)],
        eigenvectors: DenseMatrix::from_row_major_vec(
            3,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ),
        residual_norms: vec![1.0e-10, 2.0e-10],
        iterations: 2,
        converged: true,
    }
}

fn run() -> Result<(f64, f64, f64), String> {
    let contract = ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Eigen,
        |_input: &[f64]| Ok::<_, SolverError>(real_result()),
        |_forward: &TargetedEigenResult, _seed: &[f64]| {
            Ok::<_, SolverError>(real_result())
        },
        ContractValidation::<[f64], TargetedEigenResult, [f64], TargetedEigenResult>::
            dimension_checked(),
    );
    let input = vec![0.0_f64; 3];
    let forward = contract
        .solve_forward(&input)
        .map_err(|error| format!("native eigen forward failed: {error:?}"))?;
    contract
        .solve_adjoint(&forward, &input)
        .map_err(|error| format!("native eigen adjoint failed: {error:?}"))?;
    let replay = sonium_solvers::differentiable_contract::verify_forward_replay(
        &contract,
        &input,
        |left: &TargetedEigenResult, right: &TargetedEigenResult| {
            left.eigenvalues == right.eigenvalues
                && left.eigenvectors.as_slice() == right.eigenvectors.as_slice()
                && left.residual_norms == right.residual_norms
        },
    )
    .map_err(|error| format!("native eigen replay failed: {error:?}"))?;

    let malformed = TargetedEigenResult {
        eigenvalues: vec![1.0, 2.0],
        eigenvectors: DenseMatrix::from_row_major_vec(3, 1, vec![1.0, 0.0, 0.0]),
        residual_norms: vec![1.0e-10, 2.0e-10],
        iterations: 1,
        converged: true,
    };
    let malformed_contract = ClosureDifferentiableSolve::new_checked(
        DifferentiableSystemKind::Eigen,
        move |_input: &[f64]| Ok::<_, SolverError>(malformed.clone()),
        |_forward: &TargetedEigenResult, _seed: &[f64]| -> Result<TargetedEigenResult, SolverError> {
            panic!("native eigen adjoint callback must not run for malformed state")
        },
        ContractValidation::<[f64], TargetedEigenResult, [f64], TargetedEigenResult>::
            dimension_checked(),
    );
    let malformed_rejected = malformed_contract.solve_forward(&input).is_err();
    if !malformed_rejected {
        return Err("malformed native eigen state was accepted".into());
    }

    let complex = complex_result();
    complex
        .validate_contract_value("native complex eigen state")
        .map_err(|error| format!("complex native eigen validation failed: {error:?}"))?;

    Ok((
        replay.evaluations as f64,
        malformed_rejected as u8 as f64,
        complex.contract_dimension().unwrap_or_default() as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((replay_evaluations, malformed_rejected, complex_dimension)) => {
            let record = QaResult::passed(CASE_ID)
                .metric("replay_evaluations", replay_evaluations)
                .metric("malformed_rejected", malformed_rejected)
                .metric("complex_native_dimension", complex_dimension)
                .minimum("replay_evaluations", 2.0)
                .minimum("malformed_rejected", 1.0)
                .minimum("complex_native_dimension", 3.0)
                .reference("native real/complex eigen result differentiable validation and replay")
                .input_checksum("analytic/deterministic-fixture:differentiable-native-eigen-v1")
                .reference_checksum("analytic/deterministic-fixture:differentiable-native-eigen-v1")
                .dimension("3-row, 2-mode real and Complex64 eigen states")
                .tolerance(
                    "two deterministic replay evaluations; malformed state rejected; dimension 3",
                )
                .seed("solvers-differentiable-native-eigen:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .reference("native real/complex eigen result differentiable validation and replay")
                .input_checksum("analytic/deterministic-fixture:differentiable-native-eigen-v1")
                .reference_checksum("analytic/deterministic-fixture:differentiable-native-eigen-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            eprintln!("FAILED: {error}");
            println!("QA_RESULT: {}", record.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
