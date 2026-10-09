//! CPU QA for physical H(div) divergence-divergence cross-space assembly.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_two_spaces_checked};
use sonium_fem::forms::DivergenceDivergenceForm;
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};

const CASE_ID: &str = "fem.hdiv-divergence-cross-space-assembly.v1";

fn entries(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut result = BTreeMap::new();
    for index in 0..matrix.nnz() {
        *result
            .entry((matrix.rows[index], matrix.cols[index]))
            .or_insert(0.0) += matrix.values[index];
    }
    result
}

fn run() -> Result<(usize, usize, f64, usize, usize), String> {
    let mesh = unit_square_triangles(2);
    let rt0 = HdivSpace::new();
    let rt1 = HdivSpace::with_order(HdivOrder::Rt1);
    let forward =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt0, &rt1, &mesh, 8)
            .map_err(|error| error.to_string())?;
    let reverse =
        assemble_bilinear_two_spaces_checked(&DivergenceDivergenceForm, &rt1, &rt0, &mesh, 8)
            .map_err(|error| error.to_string())?;
    let expected_forward = (rt0.num_global_dofs(&mesh), rt1.num_global_dofs(&mesh));
    let expected_reverse = (expected_forward.1, expected_forward.0);
    if (forward.num_rows, forward.num_cols) != expected_forward
        || (reverse.num_rows, reverse.num_cols) != expected_reverse
    {
        return Err(format!(
            "unexpected divergence dimensions: forward={}x{}, reverse={}x{}, expected={}x{} and {}x{}",
            forward.num_rows,
            forward.num_cols,
            reverse.num_rows,
            reverse.num_cols,
            expected_forward.0,
            expected_forward.1,
            expected_reverse.0,
            expected_reverse.1,
        ));
    }
    let forward_entries = entries(&forward);
    let reverse_entries = entries(&reverse);
    let mut keys: BTreeSet<(usize, usize)> = forward_entries.keys().copied().collect();
    for &(row, column) in reverse_entries.keys() {
        keys.insert((column, row));
    }
    let mut max_error: f64 = 0.0;
    let mut nonzero = 0;
    for (row, column) in keys {
        let left = forward_entries.get(&(row, column)).copied().unwrap_or(0.0);
        let right = reverse_entries.get(&(column, row)).copied().unwrap_or(0.0);
        if !left.is_finite() || !right.is_finite() {
            return Err("non-finite divergence entry".into());
        }
        max_error = max_error.max((left - right).abs());
        nonzero += usize::from(left.abs() > 1.0e-14 || right.abs() > 1.0e-14);
    }
    if max_error > 1.0e-12 || nonzero == 0 {
        return Err(format!(
            "invalid divergence block: transpose_error={max_error}, nonzero={nonzero}"
        ));
    }

    let mut malformed = unit_square_triangles(1);
    malformed.elements[0].nodes.pop();
    let rejection_checks = usize::from(
        assemble_bilinear_two_spaces_checked(
            &DivergenceDivergenceForm,
            &HdivSpace::new(),
            &HdivSpace::with_order(HdivOrder::Rt1),
            &malformed,
            6,
        )
        .is_err(),
    );
    if rejection_checks != 1 {
        return Err("malformed topology was not rejected".into());
    }
    Ok((
        forward.num_rows,
        forward.num_cols,
        max_error,
        nonzero,
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((rows, columns, max_error, nonzero, rejection_checks)) => QaResult::passed(CASE_ID)
            .metric("rows", rows as f64)
            .metric("columns", columns as f64)
            .metric("max_transpose_error", max_error)
            .metric("nonzero_entries", nonzero as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 1.0)
            .dimension("2-D affine triangular H(div) RT0/RT1 block")
            .formulation("checked physical divergence-divergence cross-space assembly")
            .reference("swapped-order transpose parity and finite nonzero divergence block")
            .reference_checksum(
                "deterministic-invariant/divergence-transpose:fem-hdiv-divergence-cross-space-v1",
            )
            .input_checksum("unit-square-triangles-2:hdiv-rt0-rt1-divergence:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
