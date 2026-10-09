//! CPU QA for affine mixed-order scalar H1 state transfer.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, transfer_h1_state_checked, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.h1-mixed-order-transfer.v1";

fn field_p2(point: [f64; 3]) -> f64 {
    1.0 + 2.0 * point[0] - 3.0 * point[1] + 4.0 * point[0] * point[1] + point[0] * point[0]
}

fn field_p3(point: [f64; 3]) -> f64 {
    0.5 - point[0] + 2.0 * point[1] + point[0] * point[0] * point[1]
        - 0.75 * point[1] * point[1] * point[1]
}

fn state(mesh: &Mesh, space: &H1Space, field: fn([f64; 3]) -> f64) -> Result<Vec<f64>, String> {
    space
        .dof_coordinates(mesh)
        .ok_or_else(|| "H1 space did not provide DOF coordinates".to_string())
        .map(|coordinates| coordinates.into_iter().map(field).collect())
}

fn transfer_error(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    source_values: &[f64],
    field: fn([f64; 3]) -> f64,
) -> Result<f64, String> {
    let transferred = transfer_h1_state_checked(
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        source_values,
    )
    .map_err(|error| error.to_string())?;
    let expected = state(target_mesh, target_space, field)?;
    Ok(transferred
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max))
}

fn run() -> Result<(f64, f64, usize), String> {
    let p2_source_mesh = unit_square_triangles(1);
    let p3_target_mesh = unit_square_triangles(2);
    let p2_source = H1Space::new(PolynomialDegree::P2);
    let p3_target = H1Space::new(PolynomialDegree::P3);
    let p2_values = state(&p2_source_mesh, &p2_source, field_p2)?;
    let p2_to_p3 = transfer_error(
        &p2_source_mesh,
        &p3_target_mesh,
        &p2_source,
        &p3_target,
        &p2_values,
        field_p2,
    )?;

    let p3_source_mesh = unit_square_triangles(2);
    let p2_target_mesh = unit_square_triangles(1);
    let p3_source = H1Space::new(PolynomialDegree::P3);
    let p2_target = H1Space::new(PolynomialDegree::P2);
    let p3_values = state(&p3_source_mesh, &p3_source, field_p3)?;
    let p3_to_p2 = transfer_error(
        &p3_source_mesh,
        &p2_target_mesh,
        &p3_source,
        &p2_target,
        &p3_values,
        field_p3,
    )?;

    let mut non_finite = p2_values.clone();
    non_finite[0] = f64::NAN;
    let mut rejection_checks = 0;
    if transfer_h1_state_checked(
        &p2_source_mesh,
        &p3_target_mesh,
        &p2_source,
        &p3_target,
        &non_finite,
    )
    .is_err()
    {
        rejection_checks += 1;
    }
    let mut non_affine = p2_source_mesh.clone();
    for element in &mut non_affine.elements {
        element.nodes.extend([0; 7]);
    }
    if transfer_h1_state_checked(
        &non_affine,
        &p3_target_mesh,
        &p2_source,
        &p3_target,
        &p2_values,
    )
    .is_err()
    {
        rejection_checks += 1;
    }

    Ok((p2_to_p3, p3_to_p2, rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((p2_to_p3, p3_to_p2, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("p2_to_p3_max_error", p2_to_p3)
            .maximum("p2_to_p3_max_error", 2.0e-12)
            .metric("p3_to_p2_max_error", p3_to_p2)
            .maximum("p3_to_p2_max_error", 2.0e-12)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("2-D affine triangular mixed-order H1 P2/P3 transfer")
            .formulation("global-DOF state transfer through checked source-space tabulation")
            .reference("analytic quadratic and cubic polynomial fields")
            .reference_checksum("analytic-p2-p3-polynomials:fem-h1-mixed-order-v1")
            .input_checksum("unit-square-triangles-1-to-2:h1-p2-p3-affine:v1"),
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
