//! CPU QA for explicit P2 isoparametric scalar H1 transfer.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{transfer_h1_state_checked, unit_square_triangles};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.h1-curved-p2-transfer.v1";

fn run() -> Result<(f64, usize, usize), String> {
    let mut mesh = unit_square_triangles(2).to_p2();
    let curved_midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[curved_midpoint].y += 0.08;
    let space = H1Space::new(PolynomialDegree::P2);
    let source_values: Vec<f64> = (0..space.num_global_dofs(&mesh))
        .map(|dof| 0.37 + 0.19 * dof as f64 - 0.013 * (dof * dof) as f64)
        .collect();
    let transferred = transfer_h1_state_checked(&mesh, &mesh, &space, &space, &source_values)
        .map_err(|error| error.to_string())?;
    let max_error = transferred
        .iter()
        .zip(&source_values)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let mut non_finite = source_values.clone();
    non_finite[0] = f64::NAN;
    let mut rejection_checks = 0;
    if transfer_h1_state_checked(&mesh, &mesh, &space, &space, &non_finite).is_err() {
        rejection_checks += 1;
    }

    let mut p3_geometry = unit_square_triangles(1);
    for element in &mut p3_geometry.elements {
        element.nodes.extend([0; 7]);
    }
    let affine_values = vec![0.0; space.num_global_dofs(&unit_square_triangles(1))];
    if transfer_h1_state_checked(
        &p3_geometry,
        &unit_square_triangles(1),
        &space,
        &space,
        &affine_values,
    )
    .is_err()
    {
        rejection_checks += 1;
    }

    Ok((max_error, source_values.len(), rejection_checks))
}

fn main() {
    let result = match run() {
        Ok((max_error, dofs, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("global_dofs", dofs as f64)
            .minimum("global_dofs", 1.0)
            .metric("max_curved_transfer_error", max_error)
            .maximum("max_curved_transfer_error", 2.0e-11)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("2-D explicit P2 isoparametric triangular H1 transfer")
            .formulation(
                "checked global-DOF transfer through P2 inverse containment and source tabulation",
            )
            .reference("same-mesh complete global-state identity with curved edge midpoint")
            .reference_checksum("analytic/deterministic-fixture:fem-h1-curved-p2-transfer-v1")
            .input_checksum("unit-square-triangles-2:explicit-p2-curved-h1:v1"),
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
