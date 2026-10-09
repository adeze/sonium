//! CPU QA for checked order-1/order-2 H(curl) cross-space assembly.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_two_spaces_checked};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};

const CASE_ID: &str = "fem.hcurl-cross-space-assembly.v1";

struct VectorMassForm;

impl BilinearForm<f64> for VectorMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let factor = weight * test.det_j;
        let components = test.spatial_dim.min(trial.spatial_dim);
        for row in 0..test.n_dofs {
            for column in 0..trial.n_dofs {
                let dot = (0..components)
                    .map(|component| {
                        test.vector_value(row, component) * trial.vector_value(column, component)
                    })
                    .sum::<f64>();
                local_matrix.add(row, column, factor * dot);
            }
        }
    }
}

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
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let forward = assemble_bilinear_two_spaces_checked(&VectorMassForm, &order1, &order2, &mesh, 8)
        .map_err(|error| error.to_string())?;
    let reverse = assemble_bilinear_two_spaces_checked(&VectorMassForm, &order2, &order1, &mesh, 8)
        .map_err(|error| error.to_string())?;
    let forward_entries = entries(&forward);
    let reverse_entries = entries(&reverse);
    let keys: BTreeSet<_> = forward_entries
        .keys()
        .copied()
        .chain(reverse_entries.keys().map(|&(row, column)| (column, row)))
        .collect();
    let mut max_transpose_error: f64 = 0.0;
    let mut nonzero_entries = 0;
    for (row, column) in keys {
        let left = forward_entries.get(&(row, column)).copied().unwrap_or(0.0);
        let right = reverse_entries.get(&(column, row)).copied().unwrap_or(0.0);
        if !left.is_finite() || !right.is_finite() {
            return Err("H(curl) cross-mass entries must be finite".into());
        }
        max_transpose_error = max_transpose_error.max((left - right).abs());
        if left.abs() > 1.0e-14 {
            nonzero_entries += 1;
        }
    }
    if max_transpose_error > 2.0e-12 {
        return Err(format!(
            "H(curl) cross-mass transpose error is {max_transpose_error}"
        ));
    }
    if nonzero_entries <= order1.num_global_dofs(&mesh) {
        return Err("H(curl) cross-mass unexpectedly has too few nonzero entries".into());
    }

    let mut malformed = unit_square_triangles(1);
    malformed.elements[0].nodes.pop();
    let rejection_checks = usize::from(
        assemble_bilinear_two_spaces_checked(
            &VectorMassForm,
            &HcurlSpace::new(),
            &HcurlSpace::order2(),
            &malformed,
            4,
        )
        .is_err(),
    );
    Ok((
        forward.num_rows,
        forward.num_cols,
        max_transpose_error,
        nonzero_entries,
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((rows, columns, max_transpose_error, nonzero_entries, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("rows", rows as f64)
                .minimum("rows", 1.0)
                .metric("columns", columns as f64)
                .minimum("columns", 1.0)
                .metric("max_transpose_error", max_transpose_error)
                .maximum("max_transpose_error", 2.0e-12)
                .metric("nonzero_entries", nonzero_entries as f64)
                .minimum("nonzero_entries", 1.0)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 1.0)
                .dimension("2-D affine triangular H(curl) order-1/order-2")
                .formulation("checked rectangular vector cross-mass assembly")
                .reference("swapped-order transpose parity and finite/nonzero vector mass")
                .reference_checksum(
                    "analytic/vector-mass-transpose-parity:fem-hcurl-cross-space-v1",
                )
                .input_checksum("unit-square-triangles-2:hcurl-order1-order2:v1")
        }
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
