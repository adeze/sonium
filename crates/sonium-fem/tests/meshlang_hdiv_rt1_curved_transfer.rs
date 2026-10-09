#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    Mesh, apply_hdiv_rt1_state_transfer, transfer_hdiv_rt1_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use std::collections::BTreeMap;

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HdivSpace, mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

fn interpolate_plan(target_entities: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn curved_triangle_mesh() -> Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += 0.08;
    mesh
}

fn contracted_affine_mesh() -> Mesh {
    let mut mesh = unit_square_triangles(1);
    for point in &mut mesh.nodes {
        point.x = 0.1 + 0.8 * point.x;
        point.y = 0.1 + 0.8 * point.y;
    }
    mesh
}

fn state(space: &HdivSpace, mesh: &Mesh, offset: f64) -> Vec<f64> {
    (0..space.num_global_dofs(mesh))
        .map(|index| offset + 0.13 * index as f64)
        .collect()
}

fn bridge_matches_kernel(source_mesh: &Mesh, target_mesh: &Mesh, offset: f64) -> (f64, usize) {
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_values = state(&source_space, source_mesh, offset);
    let expected = transfer_hdiv_rt1_state_checked(
        source_mesh,
        target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("checked curved RT1 transfer must succeed");
    let target_entities = entities(&target_space, target_mesh);
    let transferred = apply_hdiv_rt1_state_transfer::<&'static str, _>(
        &interpolate_plan(&target_entities),
        source_mesh,
        target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("interpolation must not invoke the callback"),
    )
    .expect("MeshLang curved RT1 bridge must succeed");

    assert_eq!(transferred.len(), expected.len());
    let max_error = target_entities
        .iter()
        .map(|(entity, &index)| (transferred[entity] - expected[index]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error <= 2.0e-10,
        "curved RT1 bridge mismatch: {max_error:e}"
    );
    (max_error, target_entities.len())
}

#[test]
fn curved_rt1_bridge_matches_kernel_affine_to_curved() {
    let source = unit_square_triangles(1);
    let target = curved_triangle_mesh();
    bridge_matches_kernel(&source, &target, 0.17);
}

#[test]
fn curved_rt1_bridge_matches_kernel_curved_to_affine() {
    let source = curved_triangle_mesh();
    let target = contracted_affine_mesh();
    bridge_matches_kernel(&source, &target, 0.31);
}

/// Run both directions used by the release-facing QA binary.
pub fn meshlang_hdiv_rt1_bridge_curved_reference() -> (f64, usize) {
    let (forward_error, forward_checks) =
        bridge_matches_kernel(&unit_square_triangles(1), &curved_triangle_mesh(), 0.17);
    let (reverse_error, reverse_checks) =
        bridge_matches_kernel(&curved_triangle_mesh(), &contracted_affine_mesh(), 0.31);
    (
        forward_error.max(reverse_error),
        forward_checks + reverse_checks,
    )
}

#[test]
fn curved_rt1_bridge_rejects_nonfinite_geometry_and_state() {
    let source = unit_square_triangles(1);
    let target = curved_triangle_mesh();
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_entities = entities(&target_space, &target);
    let plan = interpolate_plan(&target_entities);

    let mut nonfinite_target = target.clone();
    nonfinite_target.nodes[0].x = f64::NAN;
    let source_values = state(&source_space, &source, 0.23);
    let error = apply_hdiv_rt1_state_transfer::<&'static str, _>(
        &plan,
        &source,
        &nonfinite_target,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("interpolation must not invoke the callback"),
    )
    .expect_err("non-finite curved geometry must be rejected");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::Interpolation(_)
    ));

    let mut nonfinite_source = source_values;
    nonfinite_source[0] = f64::INFINITY;
    let error = apply_hdiv_rt1_state_transfer::<&'static str, _>(
        &plan,
        &source,
        &target,
        &source_space,
        &target_space,
        &nonfinite_source,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("interpolation must not invoke the callback"),
    )
    .expect_err("non-finite RT1 state must be rejected");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::Interpolation(_)
    ));
}
