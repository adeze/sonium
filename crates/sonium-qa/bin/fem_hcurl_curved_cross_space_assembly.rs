//! CPU QA for curved P2 order-1/order-2 H(curl) cross-space assembly.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_two_spaces_checked};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};

const CASE_ID: &str = "fem.hcurl-curved-cross-space-assembly.v1";

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

fn curved_mesh() -> Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
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
    let mesh = curved_mesh();
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let forward =
        assemble_bilinear_two_spaces_checked(&VectorMassForm, &order1, &order2, &mesh, 10)
            .map_err(|error| error.to_string())?;
    let reverse =
        assemble_bilinear_two_spaces_checked(&VectorMassForm, &order2, &order1, &mesh, 10)
            .map_err(|error| error.to_string())?;
    let forward_entries = entries(&forward);
    let reverse_entries = entries(&reverse);
    let keys: BTreeSet<_> = forward_entries
        .keys()
        .copied()
        .chain(reverse_entries.keys().map(|&(row, column)| (column, row)))
        .collect();
    let mut max_error: f64 = 0.0;
    let mut nonzero = 0;
    for (row, column) in keys {
        let left = forward_entries.get(&(row, column)).copied().unwrap_or(0.0);
        let right = reverse_entries.get(&(column, row)).copied().unwrap_or(0.0);
        if !left.is_finite() || !right.is_finite() {
            return Err("curved H(curl) mass entries must be finite".into());
        }
        max_error = max_error.max((left - right).abs());
        nonzero += usize::from(left.abs() > 1.0e-14);
    }
    if max_error > 2.0e-10 {
        return Err(format!("curved transpose parity error is {max_error}"));
    }
    if nonzero <= order1.num_global_dofs(&mesh) {
        return Err("curved H(curl) block has too few nonzero entries".into());
    }
    let mut malformed = curved_mesh();
    malformed.elements[0].nodes.pop();
    let rejection_checks = usize::from(
        assemble_bilinear_two_spaces_checked(
            &VectorMassForm,
            &HcurlSpace::new(),
            &HcurlSpace::order2(),
            &malformed,
            6,
        )
        .is_err(),
    );
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
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("rows", rows as f64)
            .minimum("rows", 1.0)
            .metric("columns", columns as f64)
            .minimum("columns", 1.0)
            .metric("max_transpose_error", max_error)
            .maximum("max_transpose_error", 2.0e-10)
            .metric("nonzero_entries", nonzero as f64)
            .minimum("nonzero_entries", 1.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 1.0)
            .dimension("2-D explicit curved P2 triangular H(curl) order-1/order-2")
            .formulation("checked curved vector cross-mass assembly")
            .reference("swapped-order transpose parity and finite/nonzero curved vector mass")
            .reference_checksum(
                "analytic/curved-vector-mass-transpose:fem-hcurl-curved-cross-space-v1",
            )
            .input_checksum("unit-square-triangles-1:explicit-p2-curved-hcurl:v1"),
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
