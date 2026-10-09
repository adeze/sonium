//! Independent CPU evidence for the exact diagonal-Jacobi AMG adjoint cell.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_core::{Preconditioner, traits::AdjointCapability};
use sonium_solvers_preconditioners::{
    AmgConfig, AmgCycle, AmgPreconditioner, AmgRestriction, AmgSmoother,
};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.amg-diagonal-adjoint.v1";

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        8,
        8,
        (0..=8).collect(),
        (0..8).collect(),
        (0..8)
            .map(|index| Complex64::new(2.0 + index as f64, 0.0))
            .collect(),
    )
}

fn dotc(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.conj() * right)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value)
}

fn run() -> Result<(f64, f64), String> {
    let config = AmgConfig {
        cycle: AmgCycle::VCycle,
        restriction: AmgRestriction::ConjugateTranspose,
        smoother: AmgSmoother::Jacobi,
        num_pre_smooth: 2,
        num_post_smooth: 2,
        ..AmgConfig::default()
    };
    let amg = AmgPreconditioner::try_from_csr(&matrix(), config)
        .map_err(|error| format!("diagonal AMG setup failed: {error}"))?;
    if amg.transpose_capability() != AdjointCapability::Exact
        || amg.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("diagonal Jacobi AMG did not advertise exact adjoints".into());
    }

    let r: Vec<_> = (0..8)
        .map(|index| Complex64::new(index as f64 + 0.25, -0.5 * index as f64))
        .collect();
    let s: Vec<_> = (0..8)
        .map(|index| Complex64::new(0.5 * index as f64, index as f64 + 1.0))
        .collect();
    let forward_r = amg
        .try_apply(&r)
        .map_err(|error| format!("forward: {error}"))?;
    let forward_s = amg
        .try_apply(&s)
        .map_err(|error| format!("forward: {error}"))?;
    let transpose_s = amg
        .try_apply_transpose(&s)
        .map_err(|error| format!("transpose: {error}"))?;
    let hermitian_s = amg
        .try_apply_hermitian(&s)
        .map_err(|error| format!("Hermitian: {error}"))?;
    let action_error = transpose_s
        .iter()
        .zip(&forward_s)
        .chain(hermitian_s.iter().zip(&forward_s))
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0_f64, f64::max);
    let inner_product_error = (dotc(&r, &hermitian_s) - dotc(&forward_r, &s)).norm();
    Ok((action_error, inner_product_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((action_error, inner_product_error)) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent Complex64 diagonal-Jacobi AMG transpose/Hermitian action")
                .input_checksum("analytic/deterministic-fixture:solvers-amg-diagonal-adjoint-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-amg-diagonal-adjoint-v1",
                )
                .dimension("8x8 real diagonal CSR; V-cycle, equal Jacobi pre/post sweeps")
                .metric("exact_capability_checks", 2.0)
                .metric("action_error", action_error)
                .metric("hermitian_inner_product_error", inner_product_error)
                .minimum("exact_capability_checks", 2.0)
                .maximum("action_error", 1.0e-12)
                .maximum("hermitian_inner_product_error", 1.0e-12)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference("independent Complex64 diagonal-Jacobi AMG transpose/Hermitian action")
                .input_checksum("analytic/deterministic-fixture:solvers-amg-diagonal-adjoint-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-amg-diagonal-adjoint-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
