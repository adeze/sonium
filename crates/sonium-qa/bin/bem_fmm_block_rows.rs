//! Checked SLFMM/MLFMM multi-RHS row-action contract evidence.

use num_complex::Complex64;
use sonium_bem::core::assembly::{MlfmmSystem, SlfmmSystem};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.fmm-block-row-actions.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        let slfmm = SlfmmSystem::new(2, 0, 0, 0);
        let mlfmm = MlfmmSystem::new(2, 0, Vec::new(), Vec::new());
        let mut input = DenseMatrix::zeros(2, 2);
        input[[0, 0]] = Complex64::new(1.0, 0.2);
        input[[1, 0]] = Complex64::new(-2.0, 0.5);
        input[[0, 1]] = Complex64::new(0.4, -0.3);
        input[[1, 1]] = Complex64::new(0.8, 0.1);
        let rows = [1, 0, 1];
        let slfmm_forward = slfmm
            .apply_rows_block_checked(&input, &rows)
            .expect("SLFMM block");
        let slfmm_transpose = slfmm
            .apply_transpose_rows_block_checked(&input, &rows)
            .expect("SLFMM transpose block");
        let mlfmm_forward = mlfmm
            .apply_rows_block_checked(&input, &rows)
            .expect("MLFMM block");
        let mlfmm_transpose = mlfmm
            .apply_transpose_rows_block_checked(&input, &rows)
            .expect("MLFMM transpose block");

        let mut max_error: f64 = 0.0;
        for column in 0..input.ncols() {
            let vector = (0..input.nrows())
                .map(|row| input[[row, column]])
                .collect::<Vec<_>>();
            let slfmm_forward_single = slfmm
                .apply_rows_checked(&vector, &rows)
                .expect("SLFMM single");
            let slfmm_transpose_single = slfmm
                .apply_transpose_rows_checked(&vector, &rows)
                .expect("SLFMM transpose single");
            let mlfmm_forward_single = mlfmm
                .apply_rows_checked(&vector, &rows)
                .expect("MLFMM single");
            let mlfmm_transpose_single = mlfmm
                .apply_transpose_rows_checked(&vector, &rows)
                .expect("MLFMM transpose single");
            for row in 0..rows.len() {
                max_error = max_error
                    .max((slfmm_forward[[row, column]] - slfmm_forward_single[row]).norm());
                max_error = max_error
                    .max((slfmm_transpose[[row, column]] - slfmm_transpose_single[row]).norm());
                max_error = max_error
                    .max((mlfmm_forward[[row, column]] - mlfmm_forward_single[row]).norm());
                max_error = max_error
                    .max((mlfmm_transpose[[row, column]] - mlfmm_transpose_single[row]).norm());
            }
        }
        assert!(max_error == 0.0);
        let wrong = DenseMatrix::zeros(1, 2);
        assert!(slfmm.apply_rows_block_checked(&wrong, &rows).is_err());
        assert!(
            mlfmm
                .apply_transpose_rows_block_checked(&wrong, &rows)
                .is_err()
        );
        max_error
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok(max_error) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("fmm_backend_count", 2.0)
            .metric("rhs_count", 2.0)
            .metric("max_block_action_error", max_error)
            .maximum("max_block_action_error", 0.0)
            .metric("typed_dimension_rejection", 1.0)
            .minimum("typed_dimension_rejection", 1.0)
            .input_checksum("synthetic/deterministic-fixture:bem-fmm-block-row-actions-v1")
            .reference("single-RHS row projections from empty deterministic SLFMM/MLFMM systems")
            .reference_checksum("synthetic/deterministic-fixture:bem-fmm-block-row-actions-v1")
            .dimension("SLFMM and MLFMM; two DOFs; two RHS columns; duplicate row order")
            .formulation("checked local forward/transpose FMM block-row ownership contract")
            .tolerance("block actions match single-RHS projections exactly")
            .seed("bem-fmm-block-row-actions:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("FMM block-row contract panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
    };
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
