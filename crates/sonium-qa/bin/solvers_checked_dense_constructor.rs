//! Release-backed checked dense-to-CSR constructor reference.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.checked-dense-constructor.v1";

fn run() -> Result<(), String> {
    let mut nonfinite = DenseMatrix::from_elem(2, 2, 0.0_f64);
    nonfinite[[0, 0]] = f64::NAN;
    let error = CsrMatrix::try_from_dense(&nonfinite, 0.0)
        .expect_err("checked dense constructor must reject NaN");
    if !error.to_string().contains("dense value at (0, 0)") {
        return Err(format!("unexpected non-finite error: {error}"));
    }

    let finite = DenseMatrix::from_row_major_vec(1, 1, vec![1.0_f64]);
    if CsrMatrix::try_from_dense(&finite, f64::NAN).is_ok() {
        return Err("checked dense constructor accepted a NaN threshold".into());
    }

    let complex = DenseMatrix::from_row_major_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, -1.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-3.0, 4.0),
        ],
    );
    let csr = CsrMatrix::try_from_dense(&complex, 0.5)
        .map_err(|error| format!("finite complex dense matrix rejected: {error}"))?;
    if csr.nnz() != 2
        || (csr.get(0, 0) - Complex64::new(2.0, -1.0)).norm() > 1.0e-12
        || (csr.get(1, 1) - Complex64::new(-3.0, 4.0)).norm() > 1.0e-12
    {
        return Err("complex dense thresholding did not preserve expected entries".into());
    }

    Ok(())
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checked_dense_constructor_checks", 3.0)
            .minimum("checked_dense_constructor_checks", 3.0)
            .metric("nonfinite_admission", 1.0)
            .minimum("nonfinite_admission", 1.0)
            .input_checksum("dense:finite-and-nonfinite:f64-complex64:v1")
            .reference("independent checked dense-to-CSR admission and thresholding")
            .reference_checksum(
                "analytic/deterministic-fixture:solvers-checked-dense-constructor-v1",
            )
            .dimension("2x2 real/complex dense matrices")
            .formulation("CsrMatrix::try_from_dense with typed construction errors")
            .seed("solvers-checked-dense-constructor:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(message) => QaResult::failed(CASE_ID, message)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .seed("solvers-checked-dense-constructor:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };

    let passed = matches!(result.status, sonium_qa::QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
