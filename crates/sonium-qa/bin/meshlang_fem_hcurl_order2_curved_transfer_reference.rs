//! CPU-only MeshLang-to-FEM order-2 H(curl) curved-triangle reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    InterpolationError, MeshlangTransferError, Point, apply_hcurl_order2_triangle_state_transfer,
    transfer_hcurl_order2_triangle_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hcurl-order2-curved-transfer-reference.v1";

fn dof_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn curved_triangle_mesh(midpoint_offset: f64) -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += midpoint_offset;
    mesh
}

fn state(space: &HcurlSpace, mesh: &sonium_fem::mesh::Mesh) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| 0.21 + 0.07 * index as f64 - 0.002 * (index * index) as f64)
        .collect()
}

fn entities(dofs: usize) -> BTreeMap<EntityId, usize> {
    (0..dofs)
        .map(|dof| (dof_entity(dof), dof))
        .collect::<BTreeMap<_, _>>()
}

fn plan(dofs: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..dofs)
            .map(|dof| IrStateTransferEntry {
                entity: dof_entity(dof),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn bridge(
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
    source_values: &[f64],
) -> Result<Vec<f64>, MeshlangTransferError> {
    let source_space = HcurlSpace::order2();
    let target_space = HcurlSpace::order2();
    let source_dofs = source_space.num_global_dofs(source_mesh);
    let target_dofs = target_space.num_global_dofs(target_mesh);
    let result = apply_hcurl_order2_triangle_state_transfer::<&'static str, _>(
        &plan(target_dofs),
        source_mesh,
        target_mesh,
        &source_space,
        &target_space,
        source_values,
        &entities(source_dofs),
        &entities(target_dofs),
        |_, _, _| Err("interpolation callback must not run"),
    )?;
    Ok((0..target_dofs)
        .map(|dof| result[&dof_entity(dof)])
        .collect())
}

fn bridge_with_known_dofs(
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
    source_values: &[f64],
    source_dofs: usize,
    target_dofs: usize,
) -> Result<Vec<f64>, MeshlangTransferError> {
    let source_space = HcurlSpace::order2();
    let target_space = HcurlSpace::order2();
    let result = apply_hcurl_order2_triangle_state_transfer::<&'static str, _>(
        &plan(target_dofs),
        source_mesh,
        target_mesh,
        &source_space,
        &target_space,
        source_values,
        &entities(source_dofs),
        &entities(target_dofs),
        |_, _, _| Err("interpolation callback must not run"),
    )?;
    Ok((0..target_dofs)
        .map(|dof| result[&dof_entity(dof)])
        .collect())
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let affine = unit_square_triangles(1);
    let curved_inward = curved_triangle_mesh(0.08);
    let curved_outward = curved_triangle_mesh(-0.08);
    let space = HcurlSpace::order2();
    let affine_values = state(&space, &affine);
    let curved_values = state(&space, &curved_outward);

    let affine_to_curved = bridge(&affine, &curved_inward, &affine_values)
        .map_err(|error| format!("affine-to-curved bridge: {error}"))?;
    let curved_to_affine = bridge(&curved_outward, &affine, &curved_values)
        .map_err(|error| format!("curved-to-affine bridge: {error}"))?;
    let expected_affine_to_curved = transfer_hcurl_order2_triangle_state_checked(
        &affine,
        &curved_inward,
        &space,
        &space,
        &affine_values,
    )
    .map_err(|error| format!("affine-to-curved kernel: {error}"))?;
    let expected_curved_to_affine = transfer_hcurl_order2_triangle_state_checked(
        &curved_outward,
        &affine,
        &space,
        &space,
        &curved_values,
    )
    .map_err(|error| format!("curved-to-affine kernel: {error}"))?;

    let max_error = affine_to_curved
        .iter()
        .zip(&expected_affine_to_curved)
        .chain(curved_to_affine.iter().zip(&expected_curved_to_affine))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    if max_error > 2.0e-9 {
        return Err(format!("bridge/reference error {max_error:e}"));
    }

    let curved_dofs = space.num_global_dofs(&curved_inward);
    let mut nonfinite_geometry = curved_inward.clone();
    nonfinite_geometry.nodes[nonfinite_geometry.elements[0].nodes[3]] =
        Point::new_3d(f64::NAN, 0.0, 0.0);
    let rejection = bridge_with_known_dofs(
        &nonfinite_geometry,
        &curved_inward,
        &curved_values,
        curved_dofs,
        curved_dofs,
    );
    if !matches!(
        rejection,
        Err(MeshlangTransferError::Interpolation(
            InterpolationError::InvalidMesh { .. }
        ))
    ) {
        return Err("non-finite curved geometry was not rejected".into());
    }

    let mut nonfinite_values = curved_values.clone();
    nonfinite_values[0] = f64::NAN;
    let rejection = bridge_with_known_dofs(
        &curved_outward,
        &curved_outward,
        &nonfinite_values,
        curved_dofs,
        curved_dofs,
    );
    if !matches!(
        rejection,
        Err(MeshlangTransferError::Interpolation(
            InterpolationError::NonFiniteSource { .. }
        ))
    ) {
        return Err("non-finite source state was not rejected".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("affine_to_curved_checks", affine_to_curved.len() as f64)
        .metric("curved_to_affine_checks", curved_to_affine.len() as f64)
        .metric("max_bridge_reference_error", max_error)
        .metric("rejection_checks", 2.0)
        .minimum("affine_to_curved_checks", 1.0)
        .minimum("curved_to_affine_checks", 1.0)
        .maximum("max_bridge_reference_error", 2.0e-9)
        .input_checksum("mesh:affine-p1-curved-p2:meshlang-hcurl-order2-v1")
        .reference("curved order-2 H(curl) edge/bubble moments agree with checked kernel in both directions")
        .reference_checksum("analytic/deterministic-fixture:meshlang-fem-hcurl-order2-curved-v1")
        .dimension("2-D affine P1 and curved P2 triangular order-2 H(curl) state")
        .formulation("IrStateTransferPlan order-2 H(curl) curved FEM consumer")
        .tolerance("all MeshLang bridge values agree with checked kernel within 2e-9")
        .seed("meshlang-fem-hcurl-order2-curved-transfer-reference:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("QA_RESULT serialization")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
