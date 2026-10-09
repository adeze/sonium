//! CPU-only curved P2 RT1 H(div) state-transfer contract.

use sonium_fem::mesh::{
    InterpolationError, Point, transfer_hdiv_rt1_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt1-curved-p2-transfer.v1";

fn curved_triangle_mesh() -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
}

fn state(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.17 + 0.13 * index as f64)
        .collect()
}

fn max_abs_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn run() -> Result<(usize, f64, usize, usize), String> {
    let affine = unit_square_triangles(1);
    let curved = curved_triangle_mesh();
    let affine_space = HdivSpace::with_order(HdivOrder::Rt1);
    let curved_space = HdivSpace::with_order(HdivOrder::Rt1);
    let curved_state = state(&curved_space, &curved);

    let curved_identity = transfer_hdiv_rt1_state_checked(
        &curved,
        &curved,
        &curved_space,
        &curved_space,
        &curved_state,
    )
    .map_err(|error| error.to_string())?;
    let curved_identity_error = max_abs_error(&curved_identity, &curved_state);
    if curved_identity_error > 2.0e-8 {
        return Err(format!(
            "curved RT1 self-transfer error {curved_identity_error} exceeds 2e-8"
        ));
    }

    let affine_state = state(&affine_space, &affine);
    let affine_to_curved = transfer_hdiv_rt1_state_checked(
        &affine,
        &curved,
        &affine_space,
        &curved_space,
        &affine_state,
    )
    .map_err(|error| error.to_string())?;
    if affine_to_curved.len() != curved_space.num_global_dofs(&curved)
        || affine_to_curved.iter().any(|value| !value.is_finite())
    {
        return Err("affine-to-curved RT1 transfer was not finite and complete".into());
    }
    let affine_to_curved_checks = 2;

    let mut rejection_checks = 0;
    let mut nonfinite = curved.clone();
    nonfinite.nodes[nonfinite.elements[0].nodes[3]] = Point::new_3d(f64::NAN, 0.0, 0.0);
    if matches!(
        transfer_hdiv_rt1_state_checked(
            &nonfinite,
            &curved,
            &curved_space,
            &curved_space,
            &curved_state,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        rejection_checks += 1;
    }
    let rt0_space = HdivSpace::with_order(HdivOrder::Rt0);
    if matches!(
        transfer_hdiv_rt1_state_checked(
            &curved,
            &curved,
            &rt0_space,
            &curved_space,
            &vec![0.0; rt0_space.num_global_dofs(&curved)],
        ),
        Err(InterpolationError::InvalidMesh { .. })
    ) {
        rejection_checks += 1;
    }
    let mut nonfinite_state = curved_state;
    nonfinite_state[0] = f64::NAN;
    if matches!(
        transfer_hdiv_rt1_state_checked(
            &curved,
            &curved,
            &curved_space,
            &curved_space,
            &nonfinite_state,
        ),
        Err(InterpolationError::NonFiniteSource { .. })
    ) {
        rejection_checks += 1;
    }
    if rejection_checks != 3 {
        return Err(format!(
            "expected three rejection checks, got {rejection_checks}"
        ));
    }

    Ok((
        curved_space.num_global_dofs(&curved),
        curved_identity_error,
        affine_to_curved_checks,
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((target_dofs, curved_identity_error, affine_to_curved_checks, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("target_dofs", target_dofs as f64)
                .minimum("target_dofs", 1.0)
                .metric("curved_identity_max_abs_error", curved_identity_error)
                .maximum("curved_identity_max_abs_error", 2.0e-8)
                .metric("affine_to_curved_finite", 1.0)
                .minimum("affine_to_curved_finite", 1.0)
                .metric(
                    "affine_to_curved_recovery_checks",
                    affine_to_curved_checks as f64,
                )
                .minimum("affine_to_curved_recovery_checks", 2.0)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 3.0)
                .dimension("curved P2 triangles")
                .formulation("RT1 H(div) edge and cell moment state transfer")
                .input_checksum("unit-square-p1-to-curved-p2-triangles-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:fem-hdiv-rt1-curved-p2-transfer-v1",
                )
                .seed("fem-hdiv-rt1-curved-p2-transfer:cpu:v1")
        }
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
