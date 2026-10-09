//! Independent CPU contract for checked sparse submatrix extraction.

use sonium_qa::QaResult;
use sonium_solvers_core::traits::SolverError;
use sonium_solvers_sparse::{CsrMatrix, try_extract_submatrix};
use std::time::Instant;

const CASE_ID: &str = "solvers.sparse-submatrix-checked.v1";

fn source_matrix() -> CsrMatrix<f64> {
    // Rows contain an unsorted duplicate column (global column 2 in row 0).
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 4, 6],
        vec![2, 0, 2, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
}

fn operation_failed(result: Result<CsrMatrix<f64>, SolverError>) -> bool {
    matches!(
        result,
        Err(SolverError::OperationFailed {
            operation: "sparse submatrix extraction",
            ..
        })
    )
}

fn run() -> Result<[f64; 6], String> {
    let extracted = try_extract_submatrix(&source_matrix(), &[2, 0])
        .map_err(|error| format!("valid extraction failed: {error}"))?;
    let canonical = extracted.validate_canonical_structure().is_ok()
        && extracted.row_ptrs == [0, 2, 4]
        && extracted.col_indices == [0, 1, 0, 1];
    if !canonical {
        return Err(format!(
            "extracted CSR is not canonical: row_ptrs={:?}, cols={:?}",
            extracted.row_ptrs, extracted.col_indices
        ));
    }

    // Global row 2 becomes local row 0: [global 2 = 6, global 0 = 5].
    // Global row 0 becomes local row 1: duplicate global column 2 sums to 4,
    // followed by global column 0 with value 2.
    let duplicate_accumulated = extracted.values == [6.0, 5.0, 4.0, 2.0];
    if !duplicate_accumulated {
        return Err(format!(
            "duplicate entries were not accumulated: {:?}",
            extracted.values
        ));
    }

    let out_of_range = operation_failed(try_extract_submatrix(&source_matrix(), &[3]));
    let duplicate_selection = operation_failed(try_extract_submatrix(&source_matrix(), &[1, 1]));
    let rectangular = CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![0], vec![1.0]);
    let rectangular_rejected = operation_failed(try_extract_submatrix(&rectangular, &[0]));
    let mut non_finite = CsrMatrix::<f64>::identity(2);
    non_finite.values[0] = f64::NAN;
    let non_finite_rejected = try_extract_submatrix(&non_finite, &[0]).is_err();

    let checks = [
        canonical,
        duplicate_accumulated,
        out_of_range,
        duplicate_selection,
        rectangular_rejected,
        non_finite_rejected,
    ];
    if checks.iter().any(|passed| !passed) {
        return Err(format!("checked extraction contract failed: {checks:?}"));
    }
    Ok(checks.map(f64::from))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(
            [
                canonical,
                duplicate_accumulated,
                out_of_range,
                duplicate_selection,
                rectangular,
                non_finite,
            ],
        ) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .input_checksum("analytic/deterministic-fixture:sparse-submatrix-checked-v1")
            .reference_checksum("analytic/deterministic-fixture:sparse-submatrix-checked-v1")
            .metric("canonical_order", canonical)
            .metric("duplicate_accumulated", duplicate_accumulated)
            .metric("out_of_range_rejected", out_of_range)
            .metric("duplicate_selection_rejected", duplicate_selection)
            .metric("rectangular_rejected", rectangular)
            .metric("non_finite_rejected", non_finite)
            .minimum("canonical_order", 1.0)
            .minimum("duplicate_accumulated", 1.0)
            .minimum("out_of_range_rejected", 1.0)
            .minimum("duplicate_selection_rejected", 1.0)
            .minimum("rectangular_rejected", 1.0)
            .minimum("non_finite_rejected", 1.0),
        Err(error) => QaResult::failed(CASE_ID, error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!("{}", result.to_json_line().expect("QA JSON"));
}
