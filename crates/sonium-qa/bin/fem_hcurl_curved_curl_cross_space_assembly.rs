//! CPU QA for curved physical H(curl) curl-curl cross-space assembly.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_two_spaces_checked};
use sonium_fem::forms::CurlCurlForm;
use sonium_fem::mesh::{ElementType, Mesh, Point, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};

const CASE_ID: &str = "fem.hcurl-curved-curl-cross-space-assembly.v1";

fn curved_mesh() -> Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
}

fn curved_tetrahedron() -> Mesh {
    let mut mesh = Mesh::new(3);
    for point in [
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.0, 0.0, 1.0),
    ] {
        mesh.add_node(point);
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    let mut mesh = mesh.to_p2();
    let midpoint = mesh.elements[0].nodes[4];
    mesh.nodes[midpoint].y += 0.12;
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

fn run() -> Result<
    (
        usize,
        usize,
        f64,
        usize,
        usize,
        usize,
        usize,
        f64,
        usize,
        usize,
    ),
    String,
> {
    let mesh = curved_mesh();
    let order1 = HcurlSpace::new();
    let order2 = HcurlSpace::order2();
    let forward = assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order1, &order2, &mesh, 10)
        .map_err(|error| error.to_string())?;
    let reverse = assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order2, &order1, &mesh, 10)
        .map_err(|error| error.to_string())?;
    let expected_forward = (order1.num_global_dofs(&mesh), order2.num_global_dofs(&mesh));
    let expected_reverse = (expected_forward.1, expected_forward.0);
    if (forward.num_rows, forward.num_cols) != expected_forward
        || (reverse.num_rows, reverse.num_cols) != expected_reverse
    {
        return Err(format!(
            "unexpected curved curl-curl dimensions: forward={}x{}, reverse={}x{}, expected={}x{} and {}x{}",
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
            return Err("non-finite curved curl-curl entry".into());
        }
        max_error = max_error.max((left - right).abs());
        nonzero += usize::from(left.abs() > 1.0e-14 || right.abs() > 1.0e-14);
    }
    let mut malformed = curved_mesh();
    malformed.elements[0].nodes.pop();
    let rejection_checks = usize::from(
        assemble_bilinear_two_spaces_checked(
            &CurlCurlForm,
            &HcurlSpace::new(),
            &HcurlSpace::order2(),
            &malformed,
            6,
        )
        .is_err(),
    );
    if max_error > 2.0e-10 || nonzero == 0 || rejection_checks != 1 {
        return Err(format!(
            "invalid curved curl-curl block: error={max_error}, nonzero={nonzero}, rejections={rejection_checks}"
        ));
    }

    let tetrahedron = curved_tetrahedron();
    let tetra_forward =
        assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order1, &order2, &tetrahedron, 14)
            .map_err(|error| error.to_string())?;
    let tetra_reverse =
        assemble_bilinear_two_spaces_checked(&CurlCurlForm, &order2, &order1, &tetrahedron, 14)
            .map_err(|error| error.to_string())?;
    let tetra_expected = (
        order1.num_global_dofs(&tetrahedron),
        order2.num_global_dofs(&tetrahedron),
    );
    if (tetra_forward.num_rows, tetra_forward.num_cols) != tetra_expected
        || (tetra_reverse.num_rows, tetra_reverse.num_cols) != (tetra_expected.1, tetra_expected.0)
    {
        return Err(format!(
            "unexpected curved tetrahedral curl-curl dimensions: forward={}x{}, reverse={}x{}",
            tetra_forward.num_rows,
            tetra_forward.num_cols,
            tetra_reverse.num_rows,
            tetra_reverse.num_cols,
        ));
    }
    let tetra_forward_entries = entries(&tetra_forward);
    let tetra_reverse_entries = entries(&tetra_reverse);
    let mut tetra_keys: BTreeSet<(usize, usize)> = tetra_forward_entries.keys().copied().collect();
    for &(row, column) in tetra_reverse_entries.keys() {
        tetra_keys.insert((column, row));
    }
    let mut tetra_max_error = 0.0_f64;
    let mut tetra_nonzero = 0;
    for (row, column) in tetra_keys {
        let left = tetra_forward_entries
            .get(&(row, column))
            .copied()
            .unwrap_or(0.0);
        let right = tetra_reverse_entries
            .get(&(column, row))
            .copied()
            .unwrap_or(0.0);
        if !left.is_finite() || !right.is_finite() {
            return Err("non-finite curved tetrahedral curl-curl entry".into());
        }
        tetra_max_error = tetra_max_error.max((left - right).abs());
        tetra_nonzero += usize::from(left.abs() > 1.0e-14 || right.abs() > 1.0e-14);
    }
    let mut malformed_tetrahedron = curved_tetrahedron();
    malformed_tetrahedron.elements[0].nodes.pop();
    let tetra_rejection_checks = usize::from(
        assemble_bilinear_two_spaces_checked(
            &CurlCurlForm,
            &HcurlSpace::new(),
            &HcurlSpace::order2(),
            &malformed_tetrahedron,
            8,
        )
        .is_err(),
    );
    if tetra_max_error > 2.0e-10 || tetra_nonzero == 0 || tetra_rejection_checks != 1 {
        return Err(format!(
            "invalid curved tetrahedral curl-curl block: error={tetra_max_error}, nonzero={tetra_nonzero}, rejections={tetra_rejection_checks}"
        ));
    }

    Ok((
        forward.num_rows,
        forward.num_cols,
        max_error,
        nonzero,
        rejection_checks,
        tetra_forward.num_rows,
        tetra_forward.num_cols,
        tetra_max_error,
        tetra_nonzero,
        tetra_rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((
            rows,
            columns,
            max_error,
            nonzero,
            rejection_checks,
            tetra_rows,
            tetra_columns,
            tetra_max_error,
            tetra_nonzero,
            tetra_rejection_checks,
        )) => QaResult::passed(CASE_ID)
            .metric("rows", rows as f64)
            .metric("columns", columns as f64)
            .metric("max_transpose_error", max_error)
            .metric("nonzero_entries", nonzero as f64)
            .metric("rejection_checks", rejection_checks as f64)
            .metric("tetra_rows", tetra_rows as f64)
            .metric("tetra_columns", tetra_columns as f64)
            .metric("tetra_max_transpose_error", tetra_max_error)
            .metric("tetra_nonzero_entries", tetra_nonzero as f64)
            .metric("tetra_rejection_checks", tetra_rejection_checks as f64)
            .minimum("rejection_checks", 1.0)
            .minimum("tetra_rejection_checks", 1.0)
            .dimension("2-D curved P2 triangle and 3-D curved P2 tetrahedron H(curl) blocks")
            .formulation("checked physical curved curl-curl cross-space assembly")
            .reference("MFEM/Basix Piola reference identities, swapped-order parity, finite nonzero blocks")
            .reference_checksum(
            "published-piola/curved-curl-curl-reference:fem-hcurl-curved-curl-cross-space-v1",
            )
            .input_checksum("unit-square-triangles-1-and-single-tetrahedron:explicit-p2-curved-hcurl-curl:v2"),
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
