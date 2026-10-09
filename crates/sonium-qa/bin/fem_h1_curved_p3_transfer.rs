//! CPU QA for explicit P3 isoparametric scalar H1 transfer.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point, transfer_h1_state_checked};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.h1-curved-p3-transfer.v1";

fn explicit_p3_triangle_mesh(curved: bool) -> Mesh {
    let mut mesh = Mesh::new(2);
    let mut points = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (0.0, 2.0 / 3.0),
        (0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0),
    ];
    if curved {
        points[9].1 += 0.08;
    }
    for (x, y) in points {
        mesh.add_node(Point::new_2d(x, y));
    }
    mesh.add_element(ElementType::Triangle, (0..10).collect());
    mesh
}

fn run() -> Result<(f64, usize, usize), String> {
    let mesh = explicit_p3_triangle_mesh(true);
    let space = H1Space::new(PolynomialDegree::P3);
    let source_values: Vec<f64> = (0..space.num_global_dofs(&mesh))
        .map(|dof| 0.23 + 0.17 * dof as f64 - 0.011 * (dof * dof) as f64)
        .collect();
    let transferred = transfer_h1_state_checked(&mesh, &mesh, &space, &space, &source_values)
        .map_err(|error| error.to_string())?;
    let max_error = transferred
        .iter()
        .zip(&source_values)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let mut rejection_checks = 0;
    let mut nonfinite = source_values.clone();
    nonfinite[3] = f64::NAN;
    if transfer_h1_state_checked(&mesh, &mesh, &space, &space, &nonfinite).is_err() {
        rejection_checks += 1;
    }

    let mut unsupported = explicit_p3_triangle_mesh(false);
    unsupported.elements[0].nodes.push(unsupported.nodes.len());
    unsupported.add_node(Point::new_2d(0.25, 0.25));
    if transfer_h1_state_checked(&unsupported, &unsupported, &space, &space, &source_values)
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
            .minimum("global_dofs", 10.0)
            .metric("max_curved_transfer_error", max_error)
            .maximum("max_curved_transfer_error", 2.0e-10)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("2-D explicit P3 isoparametric triangular H1 transfer")
            .formulation(
                "checked global-DOF transfer through P3 inverse containment and source tabulation",
            )
            .reference("same-mesh complete global-state identity with curved cubic interior node")
            .reference_checksum("analytic/deterministic-fixture:fem-h1-curved-p3-transfer-v1")
            .input_checksum("explicit-p3-curved-triangle:v1"),
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
