//! Independent CPU evidence for checked matrix-free operator combinators.

use sonium_qa::QaResult;
use sonium_solvers::sparse::{CompositeOperator, CsrMatrix, FnOperator, SumOperator};
use sonium_solvers_core::traits::SolverError;
use std::time::Instant;

const CASE_ID: &str = "solvers.operator-combinator-boundaries.v1";

fn run() -> Result<(), String> {
    let a = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]);
    let bad = CsrMatrix::from_raw_parts(3, 2, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0, 1.0, 1.0]);
    if !matches!(
        SumOperator::try_new(&a, &bad),
        Err(SolverError::DimensionMismatch { .. })
    ) {
        return Err("checked sum constructor accepted mismatched shapes".into());
    }
    if !matches!(
        CompositeOperator::try_new(&a, &bad),
        Err(SolverError::DimensionMismatch { .. })
    ) {
        return Err("checked composition constructor accepted mismatched shapes".into());
    }

    let composition = CompositeOperator::try_new(&a, &a).map_err(|error| error.to_string())?;
    let mut output = vec![0.0; 2];
    let mut scratch = vec![0.0; 2];
    composition
        .try_apply_into_with_scratch(&[1.0, 1.0], &mut output, &mut scratch)
        .map_err(|error| error.to_string())?;
    if output != [4.0, 9.0] {
        return Err(format!("unexpected composed output: {output:?}"));
    }

    let non_finite_inner = FnOperator::new_into(2, 2, |_x: &[f64], y: &mut [f64]| {
        y.fill(f64::NAN);
    });
    let non_finite_composition =
        CompositeOperator::try_new(&a, &non_finite_inner).map_err(|error| error.to_string())?;
    if !matches!(
        non_finite_composition.try_apply_into_with_scratch(&[1.0, 2.0], &mut output, &mut scratch,),
        Err(SolverError::OperationFailed { .. })
    ) {
        return Err("checked composition accepted non-finite callback output".into());
    }
    let non_finite_sum =
        SumOperator::try_new(&a, &non_finite_inner).map_err(|error| error.to_string())?;
    if !matches!(
        non_finite_sum.try_apply_into(&[1.0, 2.0], &mut output),
        Err(SolverError::OperationFailed { .. })
    ) {
        return Err("checked sum accepted non-finite callback output".into());
    }

    let panicking_inner = FnOperator::new_into(2, 2, |_x: &[f64], _y: &mut [f64]| {
        panic!("synthetic callback failure");
    });
    let panicking_composition =
        CompositeOperator::try_new(&a, &panicking_inner).map_err(|error| error.to_string())?;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let panic_result =
        panicking_composition.try_apply_into_with_scratch(&[1.0, 2.0], &mut output, &mut scratch);
    std::panic::set_hook(previous_hook);
    if !matches!(panic_result, Err(SolverError::OperationFailed { .. })) {
        return Err("checked composition did not contain callback panic".into());
    }

    let transpose_non_finite = FnOperator::new_into_with_adjoint(
        2,
        2,
        |_x: &[f64], y: &mut [f64]| y.fill(0.0),
        |_x: &[f64], y: &mut [f64]| y.fill(f64::INFINITY),
        |_x: &[f64], y: &mut [f64]| y.fill(0.0),
    );
    let transpose_composition =
        CompositeOperator::try_new(&a, &transpose_non_finite).map_err(|error| error.to_string())?;
    if !matches!(
        transpose_composition.try_apply_transpose_into_with_scratch(
            &[1.0, 2.0],
            &mut output,
            &mut scratch,
        ),
        Err(SolverError::OperationFailed { .. })
    ) {
        return Err("checked transpose accepted non-finite callback output".into());
    }
    let forward_only = FnOperator::new(2, 2, |x: &[f64]| x.to_vec());
    let sum_without_transpose =
        SumOperator::try_new(&forward_only, &a).map_err(|error| error.to_string())?;
    if !matches!(
        sum_without_transpose.try_apply_transpose_into(&[1.0, 2.0], &mut output),
        Err(SolverError::UnsupportedOperatorOperation { .. })
    ) {
        return Err("checked sum accepted an undeclared transpose action".into());
    }

    let standalone_non_finite = FnOperator::new_into(2, 2, |_x: &[f64], y: &mut [f64]| {
        y.fill(f64::NAN);
    });
    if !matches!(
        standalone_non_finite.try_apply_into(&[1.0, 2.0], &mut output),
        Err(SolverError::OperationFailed { .. })
    ) {
        return Err("checked callback accepted non-finite forward output".into());
    }
    if !matches!(
        forward_only.try_apply_transpose_into(&[1.0, 2.0], &mut output),
        Err(SolverError::UnsupportedOperatorOperation { .. })
    ) {
        return Err("checked callback accepted undeclared transpose action".into());
    }
    let standalone_adjoint = FnOperator::new_into_with_adjoint(
        2,
        2,
        |_x: &[f64], y: &mut [f64]| y.fill(0.0),
        |_x: &[f64], y: &mut [f64]| y.fill(0.0),
        |_x: &[f64], y: &mut [f64]| y.fill(f64::INFINITY),
    );
    if !matches!(
        standalone_adjoint.try_apply_hermitian_into(&[1.0, 2.0], &mut output),
        Err(SolverError::OperationFailed { .. })
    ) {
        return Err("checked callback accepted non-finite Hermitian output".into());
    }
    if !matches!(
        composition.try_apply_into_with_scratch(&[1.0], &mut output, &mut scratch),
        Err(SolverError::DimensionMismatch { .. })
    ) {
        return Err("checked composition accepted a short input".into());
    }
    if !matches!(
        composition.try_apply_transpose_into_with_scratch(&[1.0], &mut output, &mut scratch,),
        Err(SolverError::DimensionMismatch { .. })
    ) {
        return Err("checked transpose accepted a short input".into());
    }
    Ok(())
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("checked_boundary_checks", 12.0)
            .minimum("checked_boundary_checks", 12.0)
            .metric("panic_compatibility_boundary", 1.0)
            .minimum("panic_compatibility_boundary", 1.0)
            .input_checksum(
                "analytic/deterministic-fixture:solvers-operator-combinator-boundaries-v1",
            )
        .reference(
            "checked shape/scratch admission, callback panic containment, finite-output and adjoint checks",
        )
            .reference_checksum(
                "analytic/deterministic-fixture:solvers-operator-combinator-boundaries-v1",
            )
            .seed("solvers-operator-combinator-boundaries:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
