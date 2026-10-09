//! Independent CPU evidence for transactional checked CSR SpMV actions.

use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, SolverError, sparse::CompactCsrMatrix};
use std::time::Instant;

const CASE_ID: &str = "solvers.csr-spmv-transactional.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:csr-spmv-transactional-v1";
const TOLERANCE: f64 = 1.0e-12;

fn checked_output_error(error: SolverError, operation: &'static str) -> Result<(), String> {
    if matches!(
        &error,
        SolverError::OperationFailed {
            operation: actual,
            reason,
        } if *actual == operation && reason.contains("output")
    ) {
        Ok(())
    } else {
        Err(format!("unexpected checked SpMV error: {error:?}"))
    }
}

fn run() -> Result<(f64, f64, f64, f64, f64), String> {
    let finite = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0_f64]);
    let mut finite_output = vec![0.0];
    finite
        .try_matvec_into_buf(&[3.0], &mut finite_output)
        .map_err(|error| format!("finite checked CSR SpMV failed: {error:?}"))?;
    let finite_forward_error = (finite_output[0] - 6.0).abs();
    if !finite_forward_error.is_finite() || finite_forward_error > TOLERANCE {
        return Err(format!(
            "finite checked CSR SpMV error {finite_forward_error} exceeds tolerance"
        ));
    }

    let overflowing = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let mut forward = vec![11.0];
    let forward_error = overflowing
        .try_matvec_into_buf(&[2.0], &mut forward)
        .expect_err("overflowing CSR SpMV must fail");
    checked_output_error(forward_error, "CSR SpMV")?;
    if forward.first().copied() != Some(11.0) {
        return Err("CSR SpMV mutated its destination on failure".into());
    }

    let mut accumulated = vec![13.0];
    let add_error = overflowing
        .try_matvec_add(&[2.0], &mut accumulated)
        .expect_err("overflowing CSR SpMV accumulation must fail");
    checked_output_error(add_error, "CSR SpMV add")?;
    if accumulated.first().copied() != Some(13.0) {
        return Err("CSR SpMV accumulation mutated its destination on failure".into());
    }

    let mut transpose = vec![17.0];
    let transpose_error = overflowing
        .try_matvec_transpose_into_buf(&[2.0], &mut transpose)
        .expect_err("overflowing CSR transpose SpMV must fail");
    checked_output_error(transpose_error, "CSR transpose SpMV")?;
    if transpose.first().copied() != Some(17.0) {
        return Err("CSR transpose SpMV mutated its destination on failure".into());
    }

    let mut hermitian = vec![19.0];
    let hermitian_error = overflowing
        .try_matvec_hermitian_into_buf(&[2.0], &mut hermitian)
        .expect_err("overflowing CSR Hermitian SpMV must fail");
    checked_output_error(hermitian_error, "CSR Hermitian SpMV")?;
    if hermitian.first().copied() != Some(19.0) {
        return Err("CSR Hermitian SpMV mutated its destination on failure".into());
    }

    let compact = CompactCsrMatrix::try_from_csr(&overflowing)
        .map_err(|error| format!("compact CSR construction failed: {error:?}"))?;
    let mut compact_forward = vec![23.0];
    let compact_forward_error = compact
        .try_matvec_into_buf(&[2.0], &mut compact_forward)
        .expect_err("overflowing compact CSR SpMV must fail");
    checked_output_error(compact_forward_error, "compact CSR SpMV")?;
    if compact_forward.first().copied() != Some(23.0) {
        return Err("compact CSR SpMV mutated its destination on failure".into());
    }

    let mut compact_transpose = vec![29.0];
    let compact_transpose_error = compact
        .try_matvec_transpose_into_buf(&[2.0], &mut compact_transpose)
        .expect_err("overflowing compact transpose SpMV must fail");
    checked_output_error(compact_transpose_error, "compact CSR transpose SpMV")?;
    if compact_transpose.first().copied() != Some(29.0) {
        return Err("compact transpose SpMV mutated its destination on failure".into());
    }

    let mut compact_hermitian = vec![31.0];
    let compact_hermitian_error = compact
        .try_matvec_hermitian_into_buf(&[2.0], &mut compact_hermitian)
        .expect_err("overflowing compact Hermitian SpMV must fail");
    checked_output_error(compact_hermitian_error, "compact CSR Hermitian SpMV")?;
    if compact_hermitian.first().copied() != Some(31.0) {
        return Err("compact Hermitian SpMV mutated its destination on failure".into());
    }

    Ok((finite_forward_error, 4.0, 3.0, 1.0, 1.0))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            finite_forward_error,
            csr_rejections,
            compact_rejections,
            csr_preserved,
            compact_preserved,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("1x1 CSR and compact-CSR checked SpMV actions")
                .tolerance(
                    "finite action error <= 1e-12; all seven overflow actions preserve output",
                )
                .metric("finite_forward_error", finite_forward_error)
                .metric("csr_transactional_rejections", csr_rejections)
                .metric("compact_transactional_rejections", compact_rejections)
                .metric("csr_destinations_preserved", csr_preserved)
                .metric("compact_destinations_preserved", compact_preserved)
                .maximum("finite_forward_error", TOLERANCE)
                .minimum("csr_transactional_rejections", 4.0)
                .minimum("compact_transactional_rejections", 3.0)
                .minimum("csr_destinations_preserved", 1.0)
                .minimum("compact_destinations_preserved", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
