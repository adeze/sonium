//! Native x86 sparse action and dispatch evidence.
//!
//! This test is intentionally integration-level: it drives the public CSR,
//! compact CSR, row-bucketed, SELL-C-sigma, NUMA, and BSR plan APIs and
//! compares them with an independent scalar CSR oracle. Run it with
//! `--nocapture` to archive the machine-readable evidence line. The same
//! binary can collect a scalar baseline with
//! `SONIUM_SPARSE_FORCE_SCALAR=1`.

#![cfg(target_arch = "x86_64")]

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{ComplexField, LinearOperator, SolverError};
use sonium_solvers_sparse::{
    CsrMatrix, SpmvKernel, SpmvPlan, SpmvPlanConfig, SpmvPlanPreference, detected_spmv_features,
    selected_spmv_kernel,
};
use std::path::Path;

const RECT_ROWS: usize = 7;
const RECT_COLS: usize = 11;
const EVIDENCE_SCHEMA: u32 = 2;
const FIXTURE_VERSION: u32 = 1;
const SCALAR_TYPES: [&str; 4] = ["f32", "f64", "complex32", "complex64"];
const LAYOUTS: [&str; 8] = [
    "Csr",
    "CompactCsrU32",
    "RowBucketedCsr",
    "SellCSigma",
    "NumaCsr",
    "Bsr2",
    "Bsr3",
    "Bsr4",
];
const OPERATIONS: [&str; 5] = ["accumulation", "block", "forward", "hermitian", "transpose"];
const CASE_CAPABILITIES: [&str; 7] = [
    "accumulation",
    "block",
    "forward",
    "hermitian",
    "independent-scalar-oracle",
    "transpose",
    "zero-rhs",
];

fn value<T: ComplexField>(re: f64, im: f64) -> T {
    T::from_re_im(
        T::Real::from_f64(re).expect("fixture real value is representable"),
        T::Real::from_f64(im).expect("fixture imaginary value is representable"),
    )
}

fn rectangular_matrix<T: ComplexField>() -> CsrMatrix<T> {
    // Empty rows, repeated columns, unsorted rows, and a deliberately
    // irregular row-length distribution are all part of the canonical
    // action fixture.
    let row_ptrs = vec![0, 0, 5, 5, 9, 14, 14, 19];
    let columns = vec![
        10, 0, 3, 3, 1, // row 1: duplicate 3
        2, 9, 1, 7, // row 3
        6, 5, 0, 10, 4, // row 4
        8, 2, 2, 10, 6, // row 6: duplicate 2
    ];
    let values = (0..columns.len())
        .map(|index| value::<T>(0.25 + index as f64 * 0.125, -0.5 + index as f64 * 0.0625))
        .collect();
    CsrMatrix::from_raw_parts(RECT_ROWS, RECT_COLS, row_ptrs, columns, values)
}

fn square_matrix<T: ComplexField>() -> CsrMatrix<T> {
    let rows = 12;
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for row in 0..rows {
        // A full block pattern makes BSR2/3/4 all viable while the
        // rectangular fixture above carries the duplicate/unsorted cases.
        for column in 0..rows {
            columns.push(column);
            values.push(value::<T>(
                0.25 + row as f64 * 0.03125 + column as f64 * 0.015625,
                -0.1 + column as f64 * 0.0078125,
            ));
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::from_raw_parts(rows, rows, row_ptrs, columns, values)
}

fn scalar_forward<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T]) -> Vec<T> {
    (0..matrix.num_rows)
        .map(|row| {
            let mut sum = T::zero();
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                sum += matrix.values[index] * x[matrix.col_indices[index]];
            }
            sum
        })
        .collect()
}

fn scalar_transpose<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], conjugate: bool) -> Vec<T> {
    let mut result = vec![T::zero(); matrix.num_cols];
    for (row, x_value) in x.iter().take(matrix.num_rows).enumerate() {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let value = if conjugate {
                matrix.values[index].conj()
            } else {
                matrix.values[index]
            };
            result[matrix.col_indices[index]] += value * *x_value;
        }
    }
    result
}

fn ulp_budget<T: ComplexField>() -> usize {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        8
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
        16
    } else {
        8
    }
}

fn dtype_label<T: ComplexField>() -> &'static str {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        "f32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        "f64"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
        "complex32"
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        "complex64"
    } else {
        std::any::type_name::<T>()
    }
}

fn matrix_checksum<T: ComplexField>(matrix: &CsrMatrix<T>) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for &index in &matrix.row_ptrs {
        hash = hash.rotate_left(7) ^ index as u64;
    }
    for &index in &matrix.col_indices {
        hash = hash.rotate_left(7) ^ index as u64;
    }
    for value in &matrix.values {
        hash = hash.rotate_left(7) ^ value.re().to_f64().unwrap_or(f64::NAN).to_bits();
        hash = hash.rotate_left(7) ^ value.im().to_f64().unwrap_or(f64::NAN).to_bits();
    }
    format!("{hash:016x}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn case_signature(case: &Value) -> Value {
    json!([
        case["dtype"],
        case["format"],
        case["matrix_rows"],
        case["matrix_cols"],
        case["matrix_nnz"],
        case["matrix_checksum"],
        case["actions"],
        case["capabilities"]
    ])
}

fn adversarial_signature(fixture: &Value) -> Value {
    let plans = fixture
        .get("plan_cases")
        .and_then(Value::as_array)
        .map(|plans| plans.iter().map(case_signature).collect::<Vec<_>>());
    json!([
        fixture["fixture"],
        fixture["dtype"],
        fixture["matrix_rows"],
        fixture["matrix_cols"],
        fixture["matrix_nnz"],
        fixture["matrix_checksum"],
        fixture["actions"],
        plans
    ])
}

fn failure_signature(failure: &Value) -> Value {
    json!([
        failure["fixture"],
        failure["action"],
        failure["error_kind"],
        failure["operation"],
        failure["destination_preserved"]
    ])
}

fn corpus_sha256(cases: &[Value], adversarial: &[Value], failures: &[Value]) -> String {
    let canonical = json!({
        "cases": cases.iter().map(case_signature).collect::<Vec<_>>(),
        "adversarial_fixtures": adversarial.iter().map(adversarial_signature).collect::<Vec<_>>(),
        "checked_failures": failures.iter().map(failure_signature).collect::<Vec<_>>()
    });
    let bytes = serde_json::to_vec(&canonical).expect("serialize canonical SS1 corpus");
    sha256_hex(&bytes)
}

fn negative_probe_ids(failures: &[Value]) -> Vec<String> {
    failures
        .iter()
        .map(|failure| {
            format!(
                "{}:{}",
                failure["fixture"].as_str().expect("probe fixture"),
                failure["action"].as_str().expect("probe action")
            )
        })
        .collect()
}

fn matrix_condition_proxy<T: ComplexField>(matrix: &CsrMatrix<T>) -> Option<f64> {
    let mut min_norm = f64::INFINITY;
    let mut max_norm = 0.0;
    for value in &matrix.values {
        let norm = value.norm().to_f64().unwrap_or(f64::NAN);
        if norm.is_finite() && norm > 0.0 {
            min_norm = min_norm.min(norm);
            max_norm = max_norm.max(norm);
        }
    }
    (min_norm.is_finite() && max_norm.is_finite()).then_some(max_norm / min_norm)
}

fn max_error<T: ComplexField>(actual: &[T], expected: &[T]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected)
                .norm()
                .to_f64()
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max)
}

fn assert_close<T: ComplexField>(actual: &[T], expected: &[T], tolerance: f64, label: &str) -> f64 {
    let error = max_error(actual, expected);
    assert!(
        error <= tolerance,
        "{label}: max error {error:.6e} exceeds {tolerance:.6e}"
    );
    assert!(
        actual.iter().all(|entry| entry.norm().is_finite()),
        "{label}: non-finite output"
    );
    error
}

fn plan_config(preference: SpmvPlanPreference) -> SpmvPlanConfig {
    SpmvPlanConfig {
        preference,
        expected_reuse: 16,
        prebake_transpose: true,
        prebake_hermitian: true,
        sell_c_sigma_chunk_size: 4,
        sell_c_sigma_window: 8,
        numa_partitions: (preference == SpmvPlanPreference::NumaCsr).then_some(3),
        ..SpmvPlanConfig::default()
    }
}

fn block_fixture<T: ComplexField>(cols: usize) -> DenseMatrix<T> {
    DenseMatrix::from_row_major_vec(
        cols,
        2,
        (0..cols * 2)
            .map(|index| value::<T>(0.2 + index as f64 * 0.07, -0.1 + index as f64 * 0.03))
            .collect(),
    )
}

fn exercise_plan<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    x: &[T],
    tolerance: f64,
    plan: &SpmvPlan<T>,
) -> Value {
    let forward_expected = scalar_forward(matrix, x);
    let transpose_input: Vec<T> = (0..matrix.num_rows)
        .map(|index| value::<T>(0.4 + index as f64 * 0.11, -0.3 + index as f64 * 0.05))
        .collect();
    let transpose_expected = scalar_transpose(matrix, &transpose_input, false);
    let hermitian_expected = scalar_transpose(matrix, &transpose_input, true);

    let forward = plan.try_apply(x).expect("checked forward action");
    let forward_error = assert_close(&forward, &forward_expected, tolerance, "forward");

    let mut accumulated = vec![value::<T>(2.0, -0.5); matrix.num_rows];
    let accumulated_expected = accumulated
        .iter()
        .zip(&forward_expected)
        .map(|(left, right)| *left + *right)
        .collect::<Vec<_>>();
    for (destination, contribution) in accumulated.iter_mut().zip(&forward) {
        *destination += *contribution;
    }
    let accumulation_error = assert_close(
        &accumulated,
        &accumulated_expected,
        tolerance,
        "accumulation",
    );

    let transpose = plan
        .try_apply_transpose(&transpose_input)
        .expect("checked transpose action");
    let transpose_error = assert_close(&transpose, &transpose_expected, tolerance, "transpose");
    let hermitian = plan
        .try_apply_hermitian(&transpose_input)
        .expect("checked Hermitian action");
    let hermitian_error = assert_close(&hermitian, &hermitian_expected, tolerance, "Hermitian");

    let mut forward_into = vec![T::zero(); matrix.num_rows];
    plan.apply_into(x, &mut forward_into);
    assert_close(&forward_into, &forward_expected, tolerance, "forward_into");

    let block_input = block_fixture::<T>(matrix.num_cols);
    let mut block_output = DenseMatrix::zeros(matrix.num_rows, 2);
    plan.try_apply_block(&block_input, &mut block_output)
        .expect("checked block action");
    let block_error = (0..2)
        .map(|column| {
            let rhs = block_input.column_to_vec(column);
            let expected = scalar_forward(matrix, &rhs);
            assert_close(
                &block_output.column_to_vec(column),
                &expected,
                tolerance,
                "block",
            )
        })
        .fold(0.0, f64::max);

    // Zero-RHS blocks are valid and must not manufacture a fake work item.
    let zero_input = DenseMatrix::<T>::zeros(matrix.num_cols, 0);
    let mut zero_output = DenseMatrix::<T>::zeros(matrix.num_rows, 0);
    plan.try_apply_block(&zero_input, &mut zero_output)
        .expect("zero-RHS block action");
    assert!(zero_output.as_slice().is_empty());

    json!({
        "dtype": dtype_label::<T>(),
        "format": format!("{:?}", plan.format()),
        "kernel": plan.report().selected_kernel.name(),
        "features": plan.report().features,
        "matrix_rows": matrix.num_rows,
        "matrix_cols": matrix.num_cols,
        "matrix_nnz": matrix.nnz(),
        "matrix_checksum": matrix_checksum(matrix),
        "matrix_condition_proxy": matrix_condition_proxy(matrix),
        "oracle": "independent_scalar_csr",
        "norm_tolerance": tolerance,
        "ulp_budget": ulp_budget::<T>(),
        "forward_max_abs": forward_error,
        "accumulation_max_abs": accumulation_error,
        "transpose_max_abs": transpose_error,
        "hermitian_max_abs": hermitian_error,
        "block_max_abs": block_error,
        "actions": ["forward", "accumulation", "transpose", "hermitian", "block"],
        "capabilities": CASE_CAPABILITIES
    })
}

fn exercise_family<T: ComplexField>(tolerance: f64) -> Vec<Value> {
    let matrix = rectangular_matrix::<T>();
    let mut x_storage = vec![T::zero()];
    x_storage
        .extend((0..matrix.num_cols).map(|index| value::<T>(-0.25 + index as f64 * 0.13, 0.15)));
    let x = &x_storage[1..]; // deliberately non-base-aligned input slice

    let direct = matrix.try_apply(x).expect("checked direct CSR action");
    let expected = scalar_forward(&matrix, x);
    assert_close(&direct, &expected, tolerance, "direct CSR");

    let mut accumulated = vec![value::<T>(2.0, -0.5); matrix.num_rows];
    let accumulated_expected = accumulated
        .iter()
        .zip(&expected)
        .map(|(left, right)| *left + *right)
        .collect::<Vec<_>>();
    matrix
        .try_matvec_add(x, &mut accumulated)
        .expect("checked CSR accumulation action");
    assert_close(
        &accumulated,
        &accumulated_expected,
        tolerance,
        "CSR accumulation",
    );

    let preferences = [
        SpmvPlanPreference::Csr,
        SpmvPlanPreference::CompactCsrU32,
        SpmvPlanPreference::RowBucketedCsr,
        SpmvPlanPreference::SellCSigma,
        SpmvPlanPreference::NumaCsr,
    ];
    preferences
        .into_iter()
        .map(|preference| {
            let plan = SpmvPlan::try_from_csr(&matrix, plan_config(preference))
                .expect("checked sparse plan construction");
            exercise_plan(&matrix, x, tolerance, &plan)
        })
        .collect()
}

fn exercise_bsr_family<T: ComplexField>(tolerance: f64) -> Vec<Value> {
    let matrix = square_matrix::<T>();
    let x: Vec<T> = (0..matrix.num_cols)
        .map(|index| value::<T>(0.1 + index as f64 * 0.09, -0.2))
        .collect();
    [
        SpmvPlanPreference::Bsr2,
        SpmvPlanPreference::Bsr3,
        SpmvPlanPreference::Bsr4,
    ]
    .into_iter()
    .map(|preference| {
        let plan = SpmvPlan::try_from_csr(&matrix, plan_config(preference))
            .expect("checked BSR plan construction");
        exercise_plan(&matrix, &x, tolerance, &plan)
    })
    .collect()
}

fn zero_matrix<T: ComplexField>() -> CsrMatrix<T> {
    CsrMatrix::from_raw_parts(5, 7, vec![0; 6], Vec::new(), Vec::new())
}

fn skewed_matrix<T: ComplexField>() -> CsrMatrix<T> {
    let rows = 8;
    let cols = 17;
    let mut row_ptrs = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..rows {
        let row_nnz = if row == 0 { 16 } else { 1 };
        for offset in 0..row_nnz {
            columns.push(if row == 0 { 16 - offset } else { row });
            values.push(value::<T>(1.0 + (row + offset) as f64 * 0.125, 0.0));
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::from_raw_parts(rows, cols, row_ptrs, columns, values)
}

fn adversarial_fixture_evidence<T: ComplexField>(tolerance: f64) -> Vec<Value> {
    let zero = zero_matrix::<T>();
    let zero_x = vec![value::<T>(1.0, -0.5); zero.num_cols];
    let zero_forward = zero.try_matvec(&zero_x).expect("zero forward action");
    let zero_transpose_x = vec![value::<T>(0.5, 0.25); zero.num_rows];
    let zero_transpose = zero
        .try_matvec_transpose(&zero_transpose_x)
        .expect("zero transpose action");
    let zero_hermitian = zero
        .try_matvec_hermitian(&zero_transpose_x)
        .expect("zero Hermitian action");
    assert!(zero_forward.iter().all(|entry| entry.norm().is_zero()));
    assert!(zero_transpose.iter().all(|entry| entry.norm().is_zero()));
    assert!(zero_hermitian.iter().all(|entry| entry.norm().is_zero()));

    let skewed = skewed_matrix::<T>();
    let skewed_x = vec![value::<T>(0.75, -0.125); skewed.num_cols];
    let skewed_expected = scalar_forward(&skewed, &skewed_x);
    let skewed_actual = skewed.try_matvec(&skewed_x).expect("skewed forward action");
    let skewed_error = assert_close(
        &skewed_actual,
        &skewed_expected,
        tolerance,
        "skewed forward",
    );

    // Exercise every non-BSR plan representation on both adversarial
    // fixtures. BSR requires a square block-compatible matrix and is covered
    // by the dedicated full-block family above.
    let preferences = [
        SpmvPlanPreference::Csr,
        SpmvPlanPreference::CompactCsrU32,
        SpmvPlanPreference::RowBucketedCsr,
        SpmvPlanPreference::SellCSigma,
        SpmvPlanPreference::NumaCsr,
    ];
    let mut plan_cases = Vec::with_capacity(preferences.len() * 2);
    for preference in preferences {
        let zero_plan = SpmvPlan::try_from_csr(&zero, plan_config(preference))
            .expect("zero adversarial plan construction");
        plan_cases.push(exercise_plan(&zero, &zero_x, tolerance, &zero_plan));

        let skewed_plan = SpmvPlan::try_from_csr(&skewed, plan_config(preference))
            .expect("skewed adversarial plan construction");
        plan_cases.push(exercise_plan(&skewed, &skewed_x, tolerance, &skewed_plan));
    }

    vec![
        json!({
            "fixture": "zero_matrix",
            "dtype": dtype_label::<T>(),
            "matrix_rows": zero.num_rows,
            "matrix_cols": zero.num_cols,
            "matrix_nnz": zero.nnz(),
            "matrix_checksum": matrix_checksum(&zero),
            "matrix_condition_proxy": Value::Null,
            "actions": ["forward", "transpose", "hermitian"],
            "max_abs": 0.0,
            "tolerance": tolerance,
            "plan_cases": plan_cases
        }),
        json!({
            "fixture": "skewed_rows",
            "dtype": dtype_label::<T>(),
            "matrix_rows": skewed.num_rows,
            "matrix_cols": skewed.num_cols,
            "matrix_nnz": skewed.nnz(),
            "matrix_checksum": matrix_checksum(&skewed),
            "matrix_condition_proxy": matrix_condition_proxy(&skewed),
            "actions": ["forward"],
            "max_abs": skewed_error,
            "tolerance": tolerance
        }),
    ]
}

fn error_operation(error: &SolverError) -> String {
    match error {
        SolverError::OperationFailed { operation, .. }
        | SolverError::Breakdown { operation, .. }
        | SolverError::UnsupportedOperation { operation, .. }
        | SolverError::UnsupportedOperatorOperation { operation, .. } => (*operation).to_string(),
        SolverError::DimensionMismatch { .. } => "dimension validation".to_string(),
        SolverError::ConvergenceError { .. } => "solver convergence".to_string(),
    }
}

fn checked_failure_record(
    fixture: &str,
    action: &str,
    error: &SolverError,
    destination_preserved: bool,
) -> Value {
    json!({
        "fixture": fixture,
        "action": action,
        "error_kind": format!("{:?}", error.kind()),
        "operation": error_operation(error),
        "destination_preserved": destination_preserved
    })
}

fn checked_failure_evidence() -> Vec<Value> {
    let mut failures = Vec::new();

    let mut malformed = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![1.0_f64]);
    malformed.row_ptrs[1] = 2;
    let original = vec![7.0_f64];
    let mut destination = original.clone();
    let error = malformed
        .try_matvec_into_buf(&[1.0], &mut destination)
        .expect_err("malformed CSR metadata must be rejected");
    assert_eq!(destination, original);
    failures.push(checked_failure_record(
        "malformed_csr_metadata",
        "forward_into",
        &error,
        destination == original,
    ));

    let finite_matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0_f64]);
    let original = vec![11.0_f64];
    let mut destination = original.clone();
    let error = finite_matrix
        .try_matvec_into_buf(&[f64::NAN], &mut destination)
        .expect_err("non-finite input must be rejected");
    assert_eq!(destination, original);
    failures.push(checked_failure_record(
        "non_finite_input",
        "forward_into",
        &error,
        destination == original,
    ));

    let overflow_matrix = CsrMatrix::from_raw_parts(1, 1, vec![0, 1], vec![0], vec![f64::MAX]);
    let mut forward = vec![13.0_f64];
    let original = forward.clone();
    let error = overflow_matrix
        .try_matvec_into_buf(&[f64::MAX], &mut forward)
        .expect_err("generated forward overflow must be rejected");
    assert_eq!(forward, original);
    failures.push(checked_failure_record(
        "generated_non_finite_output",
        "forward_into",
        &error,
        forward == original,
    ));

    let mut transpose = vec![17.0_f64];
    let original = transpose.clone();
    let error = overflow_matrix
        .try_matvec_transpose_into_buf(&[f64::MAX], &mut transpose)
        .expect_err("generated transpose overflow must be rejected");
    assert_eq!(transpose, original);
    failures.push(checked_failure_record(
        "generated_non_finite_output",
        "transpose_into",
        &error,
        transpose == original,
    ));

    let mut hermitian = vec![19.0_f64];
    let original = hermitian.clone();
    let error = overflow_matrix
        .try_matvec_hermitian_into_buf(&[f64::MAX], &mut hermitian)
        .expect_err("generated Hermitian overflow must be rejected");
    assert_eq!(hermitian, original);
    failures.push(checked_failure_record(
        "generated_non_finite_output",
        "hermitian_into",
        &error,
        hermitian == original,
    ));

    // Every public plan representation must reject a malformed destination
    // before writing to it, and must expose a typed SolverError. Use a
    // rectangular fixture for the row-oriented formats and a block-compatible
    // fixture for BSR so all advertised plan formats are represented.
    let rectangular = rectangular_matrix::<f64>();
    let square = square_matrix::<f64>();
    let plan_cases = [
        (SpmvPlanPreference::Csr, &rectangular),
        (SpmvPlanPreference::CompactCsrU32, &rectangular),
        (SpmvPlanPreference::RowBucketedCsr, &rectangular),
        (SpmvPlanPreference::SellCSigma, &rectangular),
        (SpmvPlanPreference::NumaCsr, &rectangular),
        (SpmvPlanPreference::Bsr2, &square),
        (SpmvPlanPreference::Bsr3, &square),
        (SpmvPlanPreference::Bsr4, &square),
    ];
    for (preference, matrix) in plan_cases {
        let plan = SpmvPlan::try_from_csr(matrix, plan_config(preference))
            .expect("checked plan construction for destination test");
        let input = block_fixture::<f64>(matrix.num_cols);
        let mut destination = DenseMatrix::from_row_major_vec(
            matrix.num_rows + 1,
            1,
            vec![23.0; matrix.num_rows + 1],
        );
        let original = destination.as_slice().to_vec();
        let error = plan
            .try_apply_block(&input, &mut destination)
            .expect_err("malformed plan destination must be rejected");
        assert_eq!(destination.as_slice(), original.as_slice());
        failures.push(checked_failure_record(
            &format!("malformed_{preference:?}_destination"),
            "block",
            &error,
            destination.as_slice() == original.as_slice(),
        ));
    }

    failures
}

#[test]
fn native_x86_sparse_dispatch_is_observable_and_safe() {
    let avx2 = std::is_x86_feature_detected!("avx2");
    let force_scalar = std::env::var_os("SONIUM_SPARSE_FORCE_SCALAR")
        .is_some_and(|value| value == "1" || value == "true");
    let disable_fma = std::env::var_os("SONIUM_SPARSE_DISABLE_FMA")
        .is_some_and(|value| value == "1" || value == "true");
    let fma = std::is_x86_feature_detected!("fma");
    let expected_simd_kernel = if avx2 && fma && !disable_fma && !force_scalar {
        SpmvKernel::Avx2Fma
    } else if avx2 && !force_scalar {
        SpmvKernel::Avx2
    } else {
        SpmvKernel::Scalar
    };
    assert_eq!(selected_spmv_kernel::<f64>(), expected_simd_kernel);
    assert_eq!(selected_spmv_kernel::<f32>(), expected_simd_kernel);
    assert_eq!(selected_spmv_kernel::<Complex64>(), expected_simd_kernel);
    assert_eq!(selected_spmv_kernel::<Complex32>(), SpmvKernel::Scalar);

    let features = detected_spmv_features();
    assert!(features.contains("arch=x86_64"));
    assert!(features.contains(&format!("avx2={avx2}")));
    assert!(features.contains(&format!("fma={fma}")));
    assert!(features.contains(&format!(
        "avx2_fma={}",
        avx2 && fma && !disable_fma && !force_scalar
    )));
    assert!(features.contains(&format!("disable_fma={disable_fma}")));
    assert!(features.contains(&format!("force_scalar={force_scalar}")));
}

#[test]
fn native_x86_sparse_action_matrix_evidence() {
    let avx2 = std::is_x86_feature_detected!("avx2");
    let force_scalar = std::env::var_os("SONIUM_SPARSE_FORCE_SCALAR")
        .is_some_and(|value| value == "1" || value == "true");
    let matrix = rectangular_matrix::<f64>();
    let mut cases = Vec::new();
    cases.extend(exercise_family::<f32>(5.0e-5));
    cases.extend(exercise_family::<f64>(1.0e-10));
    cases.extend(exercise_family::<Complex32>(8.0e-5));
    cases.extend(exercise_family::<Complex64>(1.0e-10));
    cases.extend(exercise_bsr_family::<f32>(8.0e-5));
    cases.extend(exercise_bsr_family::<f64>(1.0e-10));
    cases.extend(exercise_bsr_family::<Complex32>(1.5e-4));
    cases.extend(exercise_bsr_family::<Complex64>(1.0e-10));
    let mut adversarial_fixtures = Vec::new();
    adversarial_fixtures.extend(adversarial_fixture_evidence::<f32>(5.0e-5));
    adversarial_fixtures.extend(adversarial_fixture_evidence::<f64>(1.0e-10));
    adversarial_fixtures.extend(adversarial_fixture_evidence::<Complex32>(8.0e-5));
    adversarial_fixtures.extend(adversarial_fixture_evidence::<Complex64>(1.0e-10));
    let checked_failures = checked_failure_evidence();

    let status = if !avx2 {
        "unsupported_hardware"
    } else if force_scalar {
        "scalar_baseline"
    } else {
        "passed"
    };
    let corpus_sha256 = corpus_sha256(&cases, &adversarial_fixtures, &checked_failures);
    let probe_ids = negative_probe_ids(&checked_failures);
    let report = json!({
        "schema": EVIDENCE_SCHEMA,
        "artifact_kind": "sonium-solvers-sparse.ss1.native-x86",
        "fixture_version": FIXTURE_VERSION,
        "status": status,
        "target_arch": "x86_64",
        "compile_avx2": cfg!(target_feature = "avx2"),
        "compile_fma": cfg!(target_feature = "fma"),
        "avx2": avx2,
        "fma": std::is_x86_feature_detected!("fma"),
        "disable_fma": std::env::var_os("SONIUM_SPARSE_DISABLE_FMA")
            .is_some_and(|value| value == "1" || value == "true"),
        "force_scalar": force_scalar,
        "dispatch_features": detected_spmv_features(),
        "build_target_features": {
            "avx2": cfg!(target_feature = "avx2"),
            "fma": cfg!(target_feature = "fma")
        },
        "selected_kernels": {
            "f32": selected_spmv_kernel::<f32>().name(),
            "f64": selected_spmv_kernel::<f64>().name(),
            "complex32": selected_spmv_kernel::<Complex32>().name(),
            "complex64": selected_spmv_kernel::<Complex64>().name()
        },
        "matrix_rows": matrix.num_rows,
        "matrix_cols": matrix.num_cols,
        "matrix_nnz": matrix.nnz(),
        "matrix_checksum": matrix_checksum(&matrix),
        "matrix_condition_proxy": matrix_condition_proxy(&matrix),
        "case_count": cases.len(),
        "cases": cases,
        "adversarial_fixtures": adversarial_fixtures,
        "checked_failures": checked_failures,
        "oracle": "independent_scalar_csr",
        "coverage": {
            "scalar_types": SCALAR_TYPES,
            "layouts": LAYOUTS,
            "operations": OPERATIONS,
            "capabilities": [
                "independent-scalar-oracle",
                "typed-errors",
                "destination-preserving-failures",
                "zero-rhs"
            ]
        },
        "capability_contract": {
            "operations": {
                "forward": "exact",
                "accumulation": "exact",
                "transpose": "exact",
                "hermitian": "exact",
                "block": "exact"
            },
            "zero_rhs": "accepted",
            "checked_failures": "typed_solver_error",
            "destination": "preserved"
        },
        "negative_probes": {
            "schema": "sonium-solvers-sparse.ss1.negative-probes/v1",
            "count": probe_ids.len(),
            "probe_ids": probe_ids,
            "rejection": "typed_solver_error",
            "destination": "preserved"
        },
        "corpus_sha256": corpus_sha256
    });
    let json = serde_json::to_string(&report).expect("serialize sparse evidence report");
    println!("SONIUM_SPARSE_X86_EVIDENCE {json}");
    if let Some(path) = std::env::var_os("SONIUM_SPARSE_EVIDENCE_OUT") {
        // Cargo runs integration tests with the package directory as the
        // process cwd. Resolve relative artifact paths against the workspace
        // root so the workflow's subsequent jq/upload steps see the file at
        // the documented target/... location.
        let path = Path::new(&path);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../")
                .join(path)
        };
        if let Some(parent) = Path::new(&path).parent() {
            std::fs::create_dir_all(parent).expect("create sparse evidence artifact directory");
        }
        std::fs::write(path, json).expect("write sparse evidence artifact");
    }
}
