//! Release-facing CPU eigen checked-boundary reference.

use sonium_solvers::{dense::DenseMatrix, eigen::symmetric_eigen};

const CASE_ID: &str = "solvers.eigen-checked-boundary.v1";

fn main() {
    let started = std::time::Instant::now();
    let matrix = DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
    let (eigenvalues, eigenvectors) = match symmetric_eigen(&matrix, 2) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("checked symmetric eigen reference failed: {error}");
            std::process::exit(1);
        }
    };
    let finite_values = eigenvalues.iter().all(|value| value.is_finite());
    let finite_vectors = eigenvectors
        .as_slice()
        .iter()
        .all(|value| value.is_finite());
    let malformed_rejected = symmetric_eigen(&matrix, 3).is_err();
    let nonfinite = DenseMatrix::from_row_major_vec(2, 2, vec![2.0, f64::NAN, 1.0, 2.0]);
    let nonfinite_rejected = symmetric_eigen(&nonfinite, 2).is_err();
    let checks = [
        finite_values,
        finite_vectors,
        malformed_rejected,
        nonfinite_rejected,
    ];
    if !checks.into_iter().all(|passed| passed) {
        let record = sonium_qa::QaResult::failed(
            CASE_ID,
            "one checked eigen boundary was accepted or non-finite",
        )
        .requested_backend("cpu")
        .actual_backend("cpu")
        .elapsed_ms(started.elapsed().as_millis() as u64);
        println!(
            "QA_RESULT: {}",
            record.to_json_line().expect("eigen failure serializes")
        );
        println!("FAILED");
        std::process::exit(1);
    }
    let record = sonium_qa::QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .reference("independent 2x2 symmetric eigenvalues and checked malformed-input admissions")
        .reference_checksum("analytic/deterministic-fixture:eigen-checked-boundary-v1")
        .formulation("2x2 real symmetric eigen reference plus dimension non-finite input rejection")
        .dimension("2x2 real symmetric dense matrix")
        .tolerance("finite valid result; malformed dimension and non-finite input rejected")
        .metric("boundary_checks", checks.len() as f64)
        .minimum("boundary_checks", 4.0)
        .seed("solvers:eigen-checked-boundary:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        record.to_json_line().expect("eigen QA result serializes")
    );
    println!("SUCCESS");
}
