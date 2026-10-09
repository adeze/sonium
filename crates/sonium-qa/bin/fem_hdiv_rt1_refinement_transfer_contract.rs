//! CPU-only affine RT1 H(div) refinement state-transfer contract.

use sonium_fem::mesh::{
    ElementType, InterpolationError, Mesh, Point, transfer_hdiv_rt1_state_checked,
    unit_cube_tetrahedra, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt1-refinement-transfer.v1";

fn main() {
    let result = match run() {
        Ok((target_dofs, identity_error, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("target_dofs", target_dofs as f64)
            .minimum("target_dofs", 1.0)
            .metric("max_3d_identity_error", identity_error)
            .maximum("max_3d_identity_error", 2.0e-8)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 2.0)
            .dimension("affine P1 triangular/tetrahedral RT1 H(div) transfer")
            .formulation(
                "production contravariant-Piola reconstruction and oriented edge/face moments",
            )
            .input_checksum("unit-square-triangles-1-to-2-and-unit-cube-tetrahedra-1:rt1-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:fem-hdiv-rt1-refinement-transfer-v1",
            )
            .seed("fem-hdiv-rt1-refinement-transfer:cpu:v1"),
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

fn triangle_mesh(origin: [f64; 2], scale: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point {
        x: origin[0],
        y: origin[1],
        z: 0.0,
    });
    mesh.add_node(Point {
        x: origin[0] + scale,
        y: origin[1],
        z: 0.0,
    });
    mesh.add_node(Point {
        x: origin[0],
        y: origin[1] + scale,
        z: 0.0,
    });
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh
}

fn run() -> Result<(usize, f64, usize), String> {
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let source = unit_square_triangles(1);
    let target = unit_square_triangles(2);
    let source_values: Vec<_> = (0..space.num_global_dofs(&source))
        .map(|index| 0.125 + 0.25 * index as f64)
        .collect();
    let target_values =
        transfer_hdiv_rt1_state_checked(&source, &target, &space, &space, &source_values)
            .map_err(|error| format!("2-D refinement transfer failed: {error}"))?;
    if target_values.iter().any(|value| !value.is_finite()) {
        return Err("2-D refinement produced a non-finite value".into());
    }

    let source_3d = unit_cube_tetrahedra(1);
    let source_3d_values: Vec<_> = (0..space.num_global_dofs(&source_3d))
        .map(|index| -0.25 + 0.125 * index as f64)
        .collect();
    let transferred_3d =
        transfer_hdiv_rt1_state_checked(&source_3d, &source_3d, &space, &space, &source_3d_values)
            .map_err(|error| format!("3-D identity transfer failed: {error}"))?;
    let identity_error = transferred_3d
        .iter()
        .zip(&source_3d_values)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let outside_source = triangle_mesh([0.0, 0.0], 1.0);
    let outside_target = triangle_mesh([2.0, 2.0], 0.5);
    let outside_values = vec![0.0; space.num_global_dofs(&outside_source)];
    let outside_rejected = matches!(
        transfer_hdiv_rt1_state_checked(
            &outside_source,
            &outside_target,
            &space,
            &space,
            &outside_values,
        ),
        Err(InterpolationError::TargetElementOutside { .. })
    );
    let rt0 = HdivSpace::with_order(HdivOrder::Rt0);
    let wrong_values = vec![0.0; rt0.num_global_dofs(&outside_source)];
    let wrong_order_rejected = matches!(
        transfer_hdiv_rt1_state_checked(
            &outside_source,
            &outside_source,
            &rt0,
            &space,
            &wrong_values,
        ),
        Err(InterpolationError::InvalidMesh { .. })
    );
    let rejection_checks = usize::from(outside_rejected) + usize::from(wrong_order_rejected);
    if rejection_checks != 2 {
        return Err("RT1 rejection checks did not fail closed".into());
    }

    Ok((target_values.len(), identity_error, rejection_checks))
}
